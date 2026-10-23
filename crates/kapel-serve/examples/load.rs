//! A load on a server of OpenAI's API, as Smallem's simulator of serving
//! drew it: requests arrive as a Poisson process, each a prompt of 16 to 128
//! random tokens continued by 32 to 384 tokens (uniformly drawn, the end
//! tokens ignored), streamed. Prints, per rate of arrivals, the requests and
//! tokens served per second and the quantiles of the latency of the first
//! token, of the time per token, and of whole requests.
//!
//! `--rates max` sends every request at once: what the server sustains at
//! most. `--fractions 0.2,0.4,0.6,0.8` measures that first, then each
//! fraction of it.
//!
//! `cargo run --release -p kapel-serve --example load -- URL [--requests 500]
//! [--rates 5,10,max | --fractions 0.2,0.4,0.6,0.8] [--seed 1]`
//!
//! URL is the server's root, such as `http://127.0.0.1:8000`.

use std::io::{BufRead, BufReader};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

struct Options {
    url: String,
    requests: usize,
    rates: Vec<Option<f64>>,
    fractions: Vec<f64>,
    seed: u64,
}

fn parse() -> Result<Options, String> {
    let usage = "usage: load URL [--requests N] [--rates R,R,max | --fractions F,F] [--seed S]";
    let mut options = Options {
        url: String::new(),
        requests: 500,
        rates: Vec::new(),
        fractions: Vec::new(),
        seed: 1,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| usage.to_string());
        match arg.as_str() {
            "--requests" => options.requests = value()?.parse().map_err(|_| usage)?,
            "--seed" => options.seed = value()?.parse().map_err(|_| usage)?,
            "--rates" => {
                options.rates = value()?
                    .split(',')
                    .map(|r| match r {
                        "max" => Ok(None),
                        r => r.parse().map(Some).map_err(|_| usage.to_string()),
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--fractions" => {
                options.fractions = value()?
                    .split(',')
                    .map(|f| f.parse().map_err(|_| usage.to_string()))
                    .collect::<Result<_, _>>()?;
            }
            _ if options.url.is_empty() => options.url = arg.trim_end_matches('/').to_string(),
            _ => return Err(usage.into()),
        }
    }
    if options.url.is_empty() {
        return Err(usage.into());
    }
    if options.rates.is_empty() && options.fractions.is_empty() {
        options.rates = vec![None];
    }
    Ok(options)
}

/// xorshift: the same requests for the same seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn between(&mut self, low: usize, high: usize) -> usize {
        low + (self.next() % (high - low + 1) as u64) as usize
    }
}

struct Load {
    /// Seconds after the start.
    arrival: f64,
    prompt: Vec<u32>,
    tokens: usize,
}

fn draw(count: usize, rate: Option<f64>, seed: u64, vocab: usize) -> Vec<Load> {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut time = 0.0;
    (0..count)
        .map(|_| {
            if let Some(rate) = rate {
                // Exponential gaps: a Poisson process.
                time += -(1.0 - rng.uniform()).ln() / rate;
            }
            let length = rng.between(16, 128);
            Load {
                arrival: time,
                prompt: (0..length)
                    .map(|_| rng.between(1000, vocab - 1000) as u32)
                    .collect(),
                tokens: rng.between(32, 384),
            }
        })
        .collect()
}

/// What a request saw, in seconds after its arrival.
#[derive(Clone, Copy, Default)]
struct Seen {
    first: Option<f64>,
    end: Option<f64>,
    tokens: usize,
}

fn send(url: &str, model: &str, load: &Load) -> Result<Seen, String> {
    let start = Instant::now();
    let body = json!({
        "model": model,
        "prompt": load.prompt,
        "max_tokens": load.tokens,
        "temperature": 0,
        "ignore_eos": true,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let response = ureq::post(&format!("{url}/v1/completions"))
        .set("content-type", "application/json")
        .send_string(&body.to_string())
        .map_err(|e| e.to_string())?;
    let mut seen = Seen::default();
    for line in BufReader::new(response.into_reader()).lines() {
        let line = line.map_err(|e| e.to_string())?;
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let chunk: Value = serde_json::from_str(data).map_err(|e| e.to_string())?;
        if chunk.get("error").is_some() {
            return Err(chunk.to_string());
        }
        if chunk["choices"].as_array().is_some_and(|c| !c.is_empty()) {
            seen.first.get_or_insert(start.elapsed().as_secs_f64());
        }
        if let Some(n) = chunk["usage"]["completion_tokens"].as_u64() {
            seen.tokens = n as usize;
        }
    }
    seen.end = Some(start.elapsed().as_secs_f64());
    Ok(seen)
}

fn quantile(values: &mut [f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * q).round() as usize]
}

/// Serves `loads`; the requests per second it took, and prints a line.
fn run(url: &str, model: &str, label: &str, loads: Vec<Load>) -> f64 {
    let loads = Arc::new(loads);
    let start = Instant::now();
    let threads: Vec<_> = (0..loads.len())
        .map(|i| {
            let (url, model, loads) = (url.to_string(), model.to_string(), Arc::clone(&loads));
            let wait = Duration::from_secs_f64(loads[i].arrival);
            let at = start + wait;
            std::thread::spawn(move || {
                if let Some(rest) = at.checked_duration_since(Instant::now()) {
                    std::thread::sleep(rest);
                }
                send(&url, &model, &loads[i])
            })
        })
        .collect();
    let mut seen = Vec::new();
    let mut failed = 0;
    for thread in threads {
        match thread.join().expect("a request thread") {
            Ok(s) => seen.push(s),
            Err(e) => {
                failed += 1;
                if failed == 1 {
                    eprintln!("a request failed: {e}");
                }
            }
        }
    }
    let span = start.elapsed().as_secs_f64();
    let tokens: usize = seen.iter().map(|s| s.tokens).sum();
    let mut first: Vec<f64> = seen.iter().filter_map(|s| s.first).collect();
    let mut per_token: Vec<f64> = seen
        .iter()
        .filter_map(|s| Some((s.end? - s.first?) / (s.tokens.max(2) - 1) as f64))
        .collect();
    let mut total: Vec<f64> = seen.iter().filter_map(|s| s.end).collect();
    let rate = seen.len() as f64 / span;
    println!(
        "{label:>10} {rate:>8.2} {:>9.0} {:>8.0} {:>8.0} {:>8.1} {:>8.1} {:>8.2} {:>8.2} {failed:>6}",
        tokens as f64 / span,
        1e3 * quantile(&mut first, 0.5),
        1e3 * quantile(&mut first, 0.99),
        1e3 * quantile(&mut per_token, 0.5),
        1e3 * quantile(&mut per_token, 0.99),
        quantile(&mut total, 0.5),
        quantile(&mut total, 0.99),
    );
    rate
}

fn main() -> Result<(), String> {
    let options = parse()?;
    let models: Value = serde_json::from_reader(
        ureq::get(&format!("{}/v1/models", options.url))
            .call()
            .map_err(|e| e.to_string())?
            .into_reader(),
    )
    .map_err(|e| e.to_string())?;
    let model = models["data"][0]["id"]
        .as_str()
        .ok_or("no model")?
        .to_string();
    // The vocabulary's size is not in the API: 32000 is below every
    // model's that Kapel serves, and above the special tokens.
    let vocab = 32000;
    println!(
        "{} requests to {model}: prompts of 16 to 128 tokens, answers of 32 to 384",
        options.requests
    );
    println!(
        "{:>10} {:>8} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6}",
        "rate",
        "req/s",
        "tokens/s",
        "ttft p50",
        "ttft p99",
        "tpot p50",
        "tpot p99",
        "e2e p50",
        "e2e p99",
        "failed"
    );
    println!(
        "{:>10} {:>8} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "(req/s)", "", "", "(ms)", "(ms)", "(ms)", "(ms)", "(s)", "(s)"
    );
    for &rate in &options.rates {
        let label = rate.map_or("max".to_string(), |r| format!("{r}"));
        let loads = draw(options.requests, rate, options.seed, vocab);
        run(&options.url, &model, &label, loads);
    }
    if !options.fractions.is_empty() {
        let loads = draw(options.requests, None, options.seed, vocab);
        let max = run(&options.url, &model, "max", loads);
        for &f in &options.fractions {
            let rate = f * max;
            let loads = draw(options.requests, Some(rate), options.seed + 1, vocab);
            run(
                &options.url,
                &model,
                &format!("{:.0}% {rate:.1}", f * 100.0),
                loads,
            );
        }
    }
    Ok(())
}
