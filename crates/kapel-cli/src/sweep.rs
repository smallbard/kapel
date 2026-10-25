//! `kapel sweep`: a grid of runs; `kapel fit`: the laws of scale of their
//! results.

use std::path::{Path, PathBuf};
use std::process::Command;

use kapel_train::scaling::{self, LossLaw, OptimaLaws, Point};
use kapel_train::sweep::{self, Sweep, SweepOptions};

pub(crate) const USAGE: &str = "\
usage: kapel sweep SWEEP.toml [options]

Runs a grid of trainings (see The Kapel Book, chapter 17): SWEEP.toml names a
base configuration ([sweep] base) and lists [[run]]s, each a part of a
configuration laid over the base (model.set for the sizes, optimizer.lr,
training.budget...). Each run is a kapel train of its own, in DIR/NAME; when it
ends, a line of DIR/results.tsv gives its parameters, operations, steps, tokens,
validation loss (in nats per byte with sweep.bytes_per_token), speed and time.
Runs already in the table are not run again, runs interrupted resume from their
last checkpoint: the same command goes on after a stop.

options:
  --dry-run                print the size, steps, tokens and operations of each
                           run, and run none
  --parallel N             runs at once on the card (default sweep.parallel)
  --retry                  run again the runs noted failed
  --only NAME              only this run (repeatable)
";

pub(crate) const FIT_USAGE: &str = "\
usage: kapel fit RESULTS.tsv... [options]

Fits laws of scale to the results of kapel sweep (see The Kapel Book, chapter
17): for each budget of three runs or more, the parabola of the loss in ln N and
its lowest point, the best size N_opt and its tokens D_opt; the power laws of
N_opt, D_opt and the loss against the budget; and Chinchilla's formula
L(N, D) = E + A / N^α + B / D^β, with intervals from a bootstrap.

options:
  --loss COLUMN            the loss fitted: nats_per_byte when the table has it,
                           val_loss otherwise
  --non-embedding          count the parameters but the tables of tokens
  --predict C              predict the best size, tokens and loss for C
                           operations (repeatable)
  --check NAME             leave run NAME out of the fits and compare its loss to
                           the prediction (repeatable)
  --bootstrap N            draws for the intervals (default 200)
";

pub(crate) fn execute(args: &[String]) -> Result<(), String> {
    let mut path = None;
    let mut dry_run = false;
    let mut options = SweepOptions::default();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--dry-run" => dry_run = true,
            "--retry" => options.retry = true,
            "--parallel" => {
                let n = rest.next().ok_or("--parallel needs a number")?;
                options.parallel = Some(n.parse().map_err(|_| format!("--parallel: {n}"))?);
            }
            "--only" => options
                .only
                .push(rest.next().ok_or("--only needs a name")?.clone()),
            other if other.starts_with("--") => {
                return Err(format!("unknown option {other}\n\n{USAGE}"));
            }
            other if path.is_none() => path = Some(PathBuf::from(other)),
            other => return Err(format!("one sweep, got {other} too")),
        }
    }
    let path = path.ok_or(USAGE)?;
    let sweep = Sweep::load(&path).map_err(|e| e.to_string())?;
    if dry_run {
        let plans = sweep::plan(&sweep).map_err(|e| e.to_string())?;
        println!(
            "{:<20} {:>12} {:>14} {:>9} {:>14} {:>11}",
            "run", "parameters", "non-embedding", "steps", "tokens", "operations"
        );
        for p in plans {
            println!(
                "{:<20} {:>12} {:>14} {:>9} {:>14} {:>11.3e}",
                p.name, p.parameters, p.non_embedding, p.steps, p.tokens, p.flops
            );
        }
        return Ok(());
    }
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let train = move |config: &Path| {
        let mut c = Command::new(&exe);
        c.arg("train").arg(config);
        c
    };
    sweep::run(&sweep, &options, &train).map_err(|e| e.to_string())?;
    eprintln!("sweep: the table is {}", sweep.results().display());
    Ok(())
}

pub(crate) fn fit(args: &[String]) -> Result<(), String> {
    let mut files = Vec::new();
    let (mut loss_column, mut non_embedding, mut rounds) = (None, false, 200usize);
    let (mut predictions, mut checks) = (Vec::<f64>::new(), Vec::<String>::new());
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        let mut value = |name: &str| rest.next().cloned().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--loss" => loss_column = Some(value("--loss")?),
            "--non-embedding" => non_embedding = true,
            "--predict" => {
                let v = value("--predict")?;
                predictions.push(v.parse().map_err(|_| format!("--predict: {v}"))?);
            }
            "--check" => checks.push(value("--check")?),
            "--bootstrap" => {
                let v = value("--bootstrap")?;
                rounds = v.parse().map_err(|_| format!("--bootstrap: {v}"))?;
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown option {other}\n\n{FIT_USAGE}"));
            }
            other => files.push(PathBuf::from(other)),
        }
    }
    if files.is_empty() {
        return Err(FIT_USAGE.into());
    }
    let mut rows = Vec::new();
    for f in &files {
        rows.extend(sweep::read_results(f).map_err(|e| e.to_string())?);
    }
    let column = loss_column.unwrap_or_else(|| {
        let bytes = rows
            .iter()
            .any(|r| r.get("nats_per_byte").is_some_and(|v| !v.is_empty()));
        if bytes { "nats_per_byte" } else { "val_loss" }.to_string()
    });
    let number = |r: &std::collections::BTreeMap<String, String>, k: &str| {
        r.get(k).and_then(|v| v.parse::<f64>().ok())
    };
    let mut points = Vec::new();
    let mut left_out = Vec::new();
    for r in &rows {
        let name = r.get("name").cloned().unwrap_or_default();
        let parameters = number(
            r,
            if non_embedding {
                "non_embedding"
            } else {
                "parameters"
            },
        );
        match (
            parameters,
            number(r, "tokens"),
            number(r, "flops"),
            number(r, &column),
        ) {
            (Some(n), Some(d), Some(c), Some(l))
                if l.is_finite() && r.get("status").is_some_and(|s| s == "done") =>
            {
                points.push(Point {
                    name,
                    parameters: n,
                    tokens: d,
                    flops: c,
                    budget: number(r, "budget"),
                    loss: l,
                })
            }
            _ => left_out.push(name),
        }
    }
    println!(
        "{} runs, parameters counted {}, loss {column}",
        points.len(),
        if non_embedding {
            "but the tables"
        } else {
            "all"
        }
    );
    if !left_out.is_empty() {
        println!("left out, failed or incomplete: {}", left_out.join(", "));
    }
    let fitted: Vec<Point> = points
        .iter()
        .filter(|p| !checks.contains(&p.name))
        .cloned()
        .collect();
    // The curves at constant computation.
    let curves = scaling::isoflops(&fitted);
    for o in &curves {
        println!(
            "budget {:.3e}: N_opt = {:.2} M, D_opt = {:.1} M tokens, {:.1} tokens a parameter, \
             loss {:.4}{}",
            o.budget,
            o.parameters / 1e6,
            o.tokens / 1e6,
            o.tokens / o.parameters,
            o.loss,
            if o.curvature <= 0.0 {
                "  (no minimum: the parabola opens downwards)"
            } else {
                ""
            }
        );
        for r in &o.runs {
            println!(
                "    {:<20} N = {:>8.2} M  D = {:>8.1} M  loss {:.4}",
                r.name,
                r.parameters / 1e6,
                r.tokens / 1e6,
                r.loss
            );
        }
    }
    if let Some(laws) = OptimaLaws::fit(&curves) {
        let (a, ka) = laws.parameters;
        let (b, kb) = laws.tokens;
        let (l, kl) = laws.loss;
        println!(
            "N_opt = {ka:.3e} × C^{a:.3}   D_opt = {kb:.3e} × C^{b:.3}   L_opt = {kl:.4} × C^{l:.4}"
        );
        for &c in &predictions {
            let (n, d, loss) = laws.predict(c);
            println!(
                "  the curves predict for {c:.3e}: N = {:.2} M, D = {:.1} M, loss {loss:.4}",
                n / 1e6,
                d / 1e6
            );
        }
    }
    // The formula, on every run but the checks.
    let triples: Vec<(f64, f64, f64)> = fitted
        .iter()
        .map(|p| (p.parameters, p.tokens, p.loss))
        .collect();
    let Some(law) = scaling::fit_loss_law(&triples) else {
        println!("L(N, D): five runs at least");
        return Ok(());
    };
    println!(
        "L(N, D) = {:.4} + {:.4e} / N^{:.4} + {:.4e} / D^{:.4}",
        law.e, law.a, law.alpha, law.b, law.beta
    );
    let residuals: Vec<f64> = triples
        .iter()
        .map(|&(n, d, l)| (law.predict(n, d) - l).abs())
        .collect();
    println!(
        "  distance to the runs fitted: {:.4} on average, {:.4} at most",
        residuals.iter().sum::<f64>() / residuals.len() as f64,
        residuals.iter().copied().fold(0.0, f64::max)
    );
    let laws: Vec<LossLaw> = scaling::bootstrap(&triples, rounds, 0);
    let interval = |mut values: Vec<f64>| {
        if values.is_empty() {
            return (f64::NAN, f64::NAN);
        }
        (
            scaling::quantile(&mut values, 0.05),
            scaling::quantile(&mut values, 0.95),
        )
    };
    for (name, values) in [
        ("E", laws.iter().map(|l| l.e).collect::<Vec<_>>()),
        ("α", laws.iter().map(|l| l.alpha).collect()),
        ("β", laws.iter().map(|l| l.beta).collect()),
    ] {
        let (low, high) = interval(values);
        println!("  {name}: 90 % interval [{low:.4}, {high:.4}]");
    }
    // `C = k N D` for the optimum of the formula.
    let k = points
        .iter()
        .map(|p| p.flops / (p.parameters * p.tokens))
        .sum::<f64>()
        / points.len() as f64;
    for &c in &predictions {
        let (n, d) = law.optimum(c, k);
        let (low, high) = interval(
            laws.iter()
                .map(|l| {
                    let (n, d) = l.optimum(c, k);
                    l.predict(n, d)
                })
                .collect(),
        );
        println!(
            "  the formula predicts for {c:.3e}: N = {:.2} M, D = {:.1} M, loss {:.4} \
             [{low:.4}, {high:.4}]",
            n / 1e6,
            d / 1e6,
            law.predict(n, d)
        );
    }
    for name in &checks {
        let Some(p) = points.iter().find(|p| &p.name == name) else {
            println!("  check {name}: not in the table");
            continue;
        };
        let predicted = law.predict(p.parameters, p.tokens);
        let (low, high) = interval(
            laws.iter()
                .map(|l| l.predict(p.parameters, p.tokens))
                .collect(),
        );
        println!(
            "  check {name}: measured {:.4}, predicted {predicted:.4} [{low:.4}, {high:.4}], \
             {:+.4}",
            p.loss,
            p.loss - predicted
        );
    }
    Ok(())
}
