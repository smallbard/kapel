//! Grids of runs: one file lists them, `kapel sweep` runs them one after
//! the other, or several at once on one card, notes each in a table, and
//! after an interruption starts again where it was.
//!
//! ```toml
//! [sweep]
//! base = "train.toml"        # the configuration every run starts from
//! dir = "runs"               # each run in runs/NAME, the table in runs/results.tsv
//! parallel = 2               # runs at once on the card
//! bytes_per_token = 3.897    # the losses also in nats per byte
//!
//! [[run]]
//! name = "c1e16-e256"
//! model.set = { hidden_size = 256, num_attention_heads = 4, num_hidden_layers = 4 }
//! optimizer.lr = 2.95e-3
//! training.budget = 1e16
//! ```
//!
//! Each `[[run]]` is a part of a training configuration, laid over the base
//! (tables merged, values replaced); `checkpoint.dir` is the run's own
//! directory. Relative paths are taken from the base's directory, `dir` from
//! the sweep file's.
//!
//! A run is a `kapel train` of its own, its output in `NAME/train.out`.
//! When it ends, a line of `results.tsv` gives its sizes, its computation,
//! its losses and its speed, read from its `summary.json`. A run already in
//! the table is not run again; a run interrupted resumes from its last
//! checkpoint; a run that failed is noted `failed`, and runs again with
//! `--retry`. To run one again from the start, remove its line and its
//! directory.

use std::collections::{BTreeMap, VecDeque};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::Value;

use crate::config::TrainConfig;
use crate::trainer::{TrainError, model_json, step_flops, with_budget};

type Result<T> = std::result::Result<T, TrainError>;

fn fail<T>(message: impl Into<String>) -> Result<T> {
    Err(TrainError(message.into()))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    sweep: Header,
    #[serde(default)]
    run: Vec<toml::Table>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    base: PathBuf,
    #[serde(default)]
    dir: Option<PathBuf>,
    #[serde(default = "one")]
    parallel: usize,
    #[serde(default)]
    bytes_per_token: Option<f64>,
}

fn one() -> usize {
    1
}

/// One run of a grid: its name, and what it changes in the base.
#[derive(Clone, Debug, PartialEq)]
pub struct Planned {
    /// Its name, that of its directory.
    pub name: String,
    /// The part of a training configuration laid over the base.
    pub overrides: toml::Table,
}

/// A grid of runs, read from its file.
#[derive(Clone, Debug)]
pub struct Sweep {
    /// The base configuration.
    pub base: PathBuf,
    /// Where the runs and the table go.
    pub dir: PathBuf,
    /// The runs at once.
    pub parallel: usize,
    /// The bytes of text a token stands for, if known: the losses are then
    /// also given in nats per byte.
    pub bytes_per_token: Option<f64>,
    /// The runs, in the order of the file.
    pub runs: Vec<Planned>,
}

/// The directory of `file`, absolute: each run's configuration is read
/// from its own directory, so its paths may not be relative.
fn directory_of(file: &Path) -> Result<PathBuf> {
    let parent = file.parent().filter(|p| !p.as_os_str().is_empty());
    Ok(std::path::absolute(parent.unwrap_or(Path::new(".")))?)
}

/// Lays `over` on `base`: tables merged key by key, other values replaced.
fn merge(base: &mut toml::Table, over: &toml::Table) {
    for (key, value) in over {
        match (base.get_mut(key), value) {
            (Some(toml::Value::Table(b)), toml::Value::Table(o)) => merge(b, o),
            _ => {
                base.insert(key.clone(), value.clone());
            }
        }
    }
}

impl Sweep {
    /// Reads a grid, and checks the configuration of each of its runs.
    ///
    /// # Errors
    ///
    /// If the file cannot be read, a field is unknown, two runs share a
    /// name, or the configuration of a run is refused.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| TrainError(format!("{}: {e}", path.display())))?;
        let file: File =
            toml::from_str(&text).map_err(|e| TrainError(format!("{}: {e}", path.display())))?;
        let here = directory_of(path)?;
        let join = |p: &Path| {
            if p.is_relative() {
                here.join(p)
            } else {
                p.to_path_buf()
            }
        };
        let h = file.sweep;
        if h.parallel == 0 {
            return fail("sweep.parallel: at least 1");
        }
        if let Some(b) = h.bytes_per_token
            && !(b > 0.0 && b.is_finite())
        {
            return fail(format!("sweep.bytes_per_token: positive, got {b}"));
        }
        let mut runs = Vec::new();
        for mut table in file.run {
            let name = match table.remove("name") {
                Some(toml::Value::String(n)) => n,
                _ => return fail("[[run]]: each needs a name"),
            };
            let valid = !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
                && !name.starts_with('.');
            if !valid {
                return fail(format!(
                    "[[run]] {name:?}: a name of letters, digits, '.', '_' and '-'"
                ));
            }
            if runs.iter().any(|r: &Planned| r.name == name) {
                return fail(format!("[[run]] {name:?}: twice"));
            }
            runs.push(Planned {
                name,
                overrides: table,
            });
        }
        if runs.is_empty() {
            return fail(format!("{}: no [[run]]", path.display()));
        }
        let dir = join(&h.dir.unwrap_or_else(|| {
            PathBuf::from(
                path.file_stem()
                    .map_or("sweep".into(), |s| format!("{}-runs", s.to_string_lossy())),
            )
        }));
        let sweep = Self {
            base: join(&h.base),
            dir,
            parallel: h.parallel,
            bytes_per_token: h.bytes_per_token,
            runs,
        };
        for run in &sweep.runs {
            sweep.config(run)?;
        }
        Ok(sweep)
    }

    /// The directory of a run.
    pub fn run_dir(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    /// The table of results.
    pub fn results(&self) -> PathBuf {
        self.dir.join("results.tsv")
    }

    /// The training configuration of a run: the base with the run laid
    /// over it, in the run's directory, its paths resolved.
    ///
    /// # Errors
    ///
    /// If the base cannot be read, or the configuration is refused.
    pub fn config(&self, run: &Planned) -> Result<TrainConfig> {
        let text = std::fs::read_to_string(&self.base)
            .map_err(|e| TrainError(format!("{}: {e}", self.base.display())))?;
        let mut table: toml::Table = toml::from_str(&text)
            .map_err(|e| TrainError(format!("{}: {e}", self.base.display())))?;
        merge(&mut table, &run.overrides);
        let dir = self.run_dir(&run.name);
        let mut checkpoint = toml::Table::new();
        checkpoint.insert(
            "dir".into(),
            toml::Value::String(dir.to_string_lossy().into_owned()),
        );
        merge(
            &mut table,
            &toml::Table::from_iter([("checkpoint".to_string(), toml::Value::Table(checkpoint))]),
        );
        let text = toml::to_string(&table).map_err(|e| TrainError(e.to_string()))?;
        let mut config = TrainConfig::parse(&text)
            .map_err(|e| TrainError(format!("[[run]] {}: {e}", run.name)))?;
        config.resolve(self.base.parent().unwrap_or(Path::new(".")));
        Ok(config)
    }
}

/// What a run is before it runs: its size, its steps, its computation.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// Its name.
    pub name: String,
    /// Its parameters, all of them.
    pub parameters: usize,
    /// Its parameters but the tables of tokens and positions.
    pub non_embedding: usize,
    /// Its steps.
    pub steps: usize,
    /// Its tokens.
    pub tokens: u64,
    /// Its operations ([`crate::flops`]).
    pub flops: f64,
}

/// The size, steps and computation of each run, without running any.
///
/// # Errors
///
/// If a model's configuration cannot be read.
pub fn plan(sweep: &Sweep) -> Result<Vec<Plan>> {
    sweep
        .runs
        .iter()
        .map(|run| {
            let config = with_budget(&sweep.config(run)?)?;
            let model = kapel_models::Config::from_json(&model_json(&config)?)?;
            let shapes = crate::model::shapes(&model)?;
            let count = |table: bool| -> usize {
                shapes
                    .iter()
                    .filter(|(n, _)| table || !crate::model::is_table(n))
                    .map(|(_, s)| s.iter().product::<usize>())
                    .sum()
            };
            let b = &config.batch;
            let steps = config.schedule.steps;
            Ok(Plan {
                name: run.name.clone(),
                parameters: count(true),
                non_embedding: count(false),
                steps,
                tokens: (steps * b.pass() * b.accumulate()) as u64,
                flops: step_flops(&config, &model) as f64 * steps as f64,
            })
        })
        .collect()
}

/// The columns of `results.tsv`.
pub const COLUMNS: [&str; 13] = [
    "name",
    "status",
    "parameters",
    "non_embedding",
    "budget",
    "flops",
    "steps",
    "tokens",
    "val_loss",
    "nats_per_byte",
    "train_loss",
    "tokens_per_s",
    "seconds",
];

/// The lines of a table of results, the last of each name: by column.
///
/// # Errors
///
/// If the file cannot be read, or has no header.
pub fn read_results(path: &Path) -> Result<Vec<BTreeMap<String, String>>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| TrainError(format!("{}: {e}", path.display())))?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| TrainError(format!("{}: empty", path.display())))?
        .split('\t')
        .collect();
    let mut rows: Vec<BTreeMap<String, String>> = Vec::new();
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let row: BTreeMap<String, String> = header
            .iter()
            .zip(line.split('\t'))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        rows.retain(|r| r.get("name") != row.get("name"));
        rows.push(row);
    }
    Ok(rows)
}

/// How [`run`] goes.
#[derive(Clone, Debug, Default)]
pub struct SweepOptions {
    /// Run again the runs noted `failed`.
    pub retry: bool,
    /// The runs at once, instead of the file's.
    pub parallel: Option<usize>,
    /// Only these runs, if any are named.
    pub only: Vec<String>,
}

/// A run in progress.
struct Running {
    name: String,
    budget: Option<f64>,
    child: Child,
    started: Instant,
}

/// Runs the grid: each run not yet in the table, `parallel` at once, each
/// as `train(configuration)` with `--resume` (the command of `kapel
/// train`); notes each in the table as it ends. Gives the lines noted.
///
/// # Errors
///
/// If a configuration cannot be written, a command cannot start, or the
/// table cannot be written; a run that fails is noted, not an error.
pub fn run(
    sweep: &Sweep,
    options: &SweepOptions,
    train: &dyn Fn(&Path) -> Command,
) -> Result<Vec<BTreeMap<String, String>>> {
    std::fs::create_dir_all(&sweep.dir)?;
    let results = sweep.results();
    if !results.exists() {
        std::fs::write(&results, COLUMNS.join("\t") + "\n")?;
    }
    let done = read_results(&results)?;
    let status = |name: &str| {
        done.iter()
            .find(|r| r.get("name").map(String::as_str) == Some(name))
            .and_then(|r| r.get("status").cloned())
    };
    let mut queue: VecDeque<&Planned> = sweep
        .runs
        .iter()
        .filter(|r| options.only.is_empty() || options.only.contains(&r.name))
        .filter(|r| match status(&r.name).as_deref() {
            None => true,
            Some("failed") => options.retry,
            Some(_) => false,
        })
        .collect();
    let total = queue.len();
    let parallel = options.parallel.unwrap_or(sweep.parallel).max(1);
    eprintln!(
        "sweep: {total} runs to do of {}, {parallel} at once, in {}",
        sweep.runs.len(),
        sweep.dir.display()
    );
    let mut running: Vec<Running> = Vec::new();
    let mut noted = Vec::new();
    let mut started = 0;
    while !queue.is_empty() || !running.is_empty() {
        while running.len() < parallel
            && let Some(run) = queue.pop_front()
        {
            let config = with_budget(&sweep.config(run)?)?;
            let dir = sweep.run_dir(&run.name);
            std::fs::create_dir_all(&dir)?;
            // A run that ended but was not noted: noted now.
            if finished(&dir, config.schedule.steps) {
                let row = note(sweep, &run.name, config.training.budget, Ok(()), None)?;
                noted.push(row);
                continue;
            }
            let path = dir.join("train.toml");
            std::fs::write(&path, config.to_toml())?;
            let out = std::fs::File::create(dir.join("train.out"))?;
            let child = train(&path)
                .arg("--resume")
                .stdout(Stdio::from(out.try_clone()?))
                .stderr(Stdio::from(out))
                .spawn()?;
            started += 1;
            eprintln!(
                "sweep: {} started ({started} of {total}), {} steps",
                run.name, config.schedule.steps
            );
            running.push(Running {
                name: run.name.clone(),
                budget: config.training.budget,
                child,
                started: Instant::now(),
            });
        }
        std::thread::sleep(Duration::from_millis(500));
        let mut i = 0;
        while i < running.len() {
            if let Some(exit) = running[i].child.try_wait()? {
                let r = running.swap_remove(i);
                let outcome = if exit.success() {
                    Ok(())
                } else {
                    Err(format!("{exit}"))
                };
                let row = note(
                    sweep,
                    &r.name,
                    r.budget,
                    outcome,
                    Some(r.started.elapsed().as_secs_f64()),
                )?;
                noted.push(row);
            } else {
                i += 1;
            }
        }
    }
    Ok(noted)
}

/// Whether the run in `dir` has a summary of all its `steps`.
fn finished(dir: &Path, steps: usize) -> bool {
    std::fs::read(dir.join("summary.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|s| s["steps"].as_u64())
        == Some(steps as u64)
}

/// Writes the line of a run that ended, from its summary.
fn note(
    sweep: &Sweep,
    name: &str,
    budget: Option<f64>,
    outcome: std::result::Result<(), String>,
    seconds: Option<f64>,
) -> Result<BTreeMap<String, String>> {
    let dir = sweep.run_dir(name);
    let summary: Option<Value> = std::fs::read(dir.join("summary.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok());
    let failed = outcome.is_err() || summary.is_none();
    let s = summary.unwrap_or(Value::Null);
    let number = |key: &str| s[key].as_f64();
    let val = number("val_loss");
    let show =
        |v: Option<f64>, precision: usize| v.map_or(String::new(), |v| format!("{v:.precision$}"));
    let integer = |key: &str| s[key].as_u64().map_or(String::new(), |v| v.to_string());
    let mut row = BTreeMap::new();
    let mut put = |k: &str, v: String| {
        row.insert(k.to_string(), v);
    };
    put("name", name.to_string());
    put("status", if failed { "failed" } else { "done" }.to_string());
    put("parameters", integer("parameters"));
    put("non_embedding", integer("non_embedding_parameters"));
    put("budget", budget.map_or(String::new(), |b| format!("{b:e}")));
    put(
        "flops",
        number("flops").map_or(String::new(), |f| format!("{f:.4e}")),
    );
    put("steps", integer("steps"));
    put("tokens", integer("tokens"));
    put("val_loss", show(val, 4));
    put(
        "nats_per_byte",
        show(val.zip(sweep.bytes_per_token).map(|(v, b)| v / b), 4),
    );
    put("train_loss", show(number("loss"), 4));
    put("tokens_per_s", show(number("tokens_per_s"), 0));
    put("seconds", show(seconds.or_else(|| number("seconds")), 0));
    let line: Vec<&str> = COLUMNS.iter().map(|c| row[*c].as_str()).collect();
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(sweep.results())?;
    writeln!(file, "{}", line.join("\t"))?;
    match outcome {
        Err(e) if failed => eprintln!(
            "sweep: {name} failed ({e}); see {}",
            dir.join("train.out").display()
        ),
        _ if failed => eprintln!("sweep: {name} failed: no summary.json"),
        _ => eprintln!(
            "sweep: {name} done{}: validation {}{}",
            seconds.map_or(String::new(), |s| format!(" in {s:.0} s")),
            row["val_loss"],
            if row["nats_per_byte"].is_empty() {
                String::new()
            } else {
                format!(" ({} nats per byte)", row["nats_per_byte"])
            }
        ),
    }
    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_run_is_laid_over_the_base() {
        let dir = std::env::temp_dir().join(format!("kapel-sweep-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("base.toml"),
            "[model]\nconfig = \"m.json\"\n[data]\ntrain = \"t.bin\"\n\
             [optimizer]\nkind = \"muon\"\nlr = 1e-3\n[schedule]\nsteps = 10\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("grid.toml"),
            "[sweep]\nbase = \"base.toml\"\nparallel = 2\n\n\
             [[run]]\nname = \"a\"\noptimizer.lr = 2e-3\nmodel.set = { hidden_size = 64 }\n\n\
             [[run]]\nname = \"b\"\nschedule = { steps = 20 }\n",
        )
        .unwrap();
        let sweep = Sweep::load(&dir.join("grid.toml")).unwrap();
        assert_eq!(sweep.dir, dir.join("grid-runs"));
        let a = sweep.config(&sweep.runs[0]).unwrap();
        assert_eq!(a.optimizer.lr, 2e-3);
        assert_eq!(a.optimizer.kind, "muon");
        assert_eq!(a.model.set["hidden_size"].as_integer(), Some(64));
        assert_eq!(a.model.config, dir.join("m.json"));
        assert!(a.checkpoint.dir.ends_with("grid-runs/a"));
        // Paths are absolute, even from a relative sweep file: a run reads
        // its configuration from its own directory.
        let here = std::env::current_dir().unwrap();
        assert_eq!(directory_of(Path::new("grid.toml")).unwrap(), here);
        assert_eq!(
            directory_of(Path::new("a/grid.toml")).unwrap(),
            here.join("a")
        );
        assert!(
            a.data
                .train
                .as_deref()
                .is_some_and(|t| Path::new(t).is_absolute())
        );
        let b = sweep.config(&sweep.runs[1]).unwrap();
        assert_eq!((b.schedule.steps, b.optimizer.lr), (20, 1e-3));
        // A mistake in a run is found before anything runs.
        std::fs::write(
            dir.join("bad.toml"),
            "[sweep]\nbase = \"base.toml\"\n[[run]]\nname = \"c\"\noptimizer.lrr = 1\n",
        )
        .unwrap();
        let e = Sweep::load(&dir.join("bad.toml")).unwrap_err().to_string();
        assert!(e.contains("lrr"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_last_line_of_a_name_counts() {
        let path = std::env::temp_dir().join(format!("kapel-results-{}.tsv", std::process::id()));
        std::fs::write(
            &path,
            "name\tstatus\tval_loss\na\tfailed\t\nb\tdone\t3.1\na\tdone\t3.0\n",
        )
        .unwrap();
        let rows = read_results(&path).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1]["name"], "a");
        assert_eq!(rows[1]["val_loss"], "3.0");
        std::fs::remove_file(&path).unwrap();
    }
}
