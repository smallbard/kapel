//! What the server measures, in Prometheus' text format (`/metrics`).
//!
//! Counters and histograms are atomics: the handlers and the engine's
//! thread add to them without locks.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};

use kapel_infer::EngineStats;

/// The upper bounds of the buckets of the latency histograms, in seconds.
const LATENCY_BOUNDS: [f64; 16] = [
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
];

/// A histogram with fixed buckets.
pub(crate) struct Histogram {
    bounds: &'static [f64],
    /// Per bucket, the observations at most its bound (not cumulated); then
    /// those above every bound.
    counts: Vec<AtomicU64>,
    /// The sum of the observations, in microseconds.
    sum_micros: AtomicU64,
}

impl Histogram {
    fn new(bounds: &'static [f64]) -> Self {
        Self {
            bounds,
            counts: (0..=bounds.len()).map(|_| AtomicU64::new(0)).collect(),
            sum_micros: AtomicU64::new(0),
        }
    }

    pub(crate) fn observe(&self, seconds: f64) {
        let at = self
            .bounds
            .iter()
            .position(|&b| seconds <= b)
            .unwrap_or(self.bounds.len());
        self.counts[at].fetch_add(1, Ordering::Relaxed);
        self.sum_micros
            .fetch_add((seconds * 1e6) as u64, Ordering::Relaxed);
    }

    fn render(&self, out: &mut String, name: &str, help: &str) {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
        let mut total = 0;
        for (i, count) in self.counts.iter().enumerate() {
            total += count.load(Ordering::Relaxed);
            let bound = self
                .bounds
                .get(i)
                .map_or("+Inf".to_string(), f64::to_string);
            let _ = writeln!(out, "{name}_bucket{{le=\"{bound}\"}} {total}");
        }
        let sum = self.sum_micros.load(Ordering::Relaxed) as f64 / 1e6;
        let _ = writeln!(out, "{name}_sum {sum}\n{name}_count {total}");
    }
}

/// How requests ended, for `kapel_requests_total`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Outcome {
    /// Answered.
    Ok,
    /// Refused before it was queued: invalid, or too many waiting.
    Refused,
    /// The client went away first.
    Cancelled,
    /// The server failed.
    Failed,
}

/// Everything the server counts.
pub(crate) struct Metrics {
    requests: [AtomicU64; 4],
    pub(crate) prompt_tokens: AtomicU64,
    pub(crate) generated_tokens: AtomicU64,
    pub(crate) first_token: Histogram,
    pub(crate) between_tokens: Histogram,
    pub(crate) duration: Histogram,
    // Copied from the engine after each pass.
    waiting: AtomicU64,
    running: AtomicU64,
    pages: AtomicU64,
    free_pages: AtomicU64,
    preemptions: AtomicU64,
    shared_positions: AtomicU64,
    passes: AtomicU64,
    recorded_passes: AtomicU64,
}

impl Metrics {
    pub(crate) fn new() -> Self {
        Self {
            requests: Default::default(),
            prompt_tokens: AtomicU64::new(0),
            generated_tokens: AtomicU64::new(0),
            first_token: Histogram::new(&LATENCY_BOUNDS),
            between_tokens: Histogram::new(&LATENCY_BOUNDS),
            duration: Histogram::new(&LATENCY_BOUNDS),
            waiting: AtomicU64::new(0),
            running: AtomicU64::new(0),
            pages: AtomicU64::new(0),
            free_pages: AtomicU64::new(0),
            preemptions: AtomicU64::new(0),
            shared_positions: AtomicU64::new(0),
            passes: AtomicU64::new(0),
            recorded_passes: AtomicU64::new(0),
        }
    }

    pub(crate) fn ended(&self, outcome: Outcome) {
        self.requests[outcome as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// Notes what the engine is doing.
    pub(crate) fn engine(&self, stats: &EngineStats) {
        let set = |a: &AtomicU64, v: u64| a.store(v, Ordering::Relaxed);
        set(&self.waiting, stats.waiting as u64);
        set(&self.running, stats.running as u64);
        set(&self.pages, stats.pages as u64);
        set(&self.free_pages, stats.free_pages as u64);
        set(&self.preemptions, stats.preemptions);
        set(&self.shared_positions, stats.shared_positions);
        set(&self.passes, stats.passes);
        set(&self.recorded_passes, stats.recorded_passes);
    }

    /// The text of `/metrics`.
    pub(crate) fn render(&self) -> String {
        let mut out = String::new();
        let get = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let _ = writeln!(
            out,
            "# HELP kapel_requests_total Requests, by how they ended.\n\
             # TYPE kapel_requests_total counter"
        );
        for (outcome, count) in ["ok", "refused", "cancelled", "failed"]
            .iter()
            .zip(&self.requests)
        {
            let _ = writeln!(
                out,
                "kapel_requests_total{{outcome=\"{outcome}\"}} {}",
                get(count)
            );
        }
        let mut counter = |name: &str, help: &str, value: u64| {
            let _ = writeln!(
                out,
                "# HELP {name} {help}\n# TYPE {name} counter\n{name} {value}"
            );
        };
        counter(
            "kapel_prompt_tokens_total",
            "Prompt tokens of the requests admitted.",
            get(&self.prompt_tokens),
        );
        counter(
            "kapel_generation_tokens_total",
            "Tokens generated.",
            get(&self.generated_tokens),
        );
        counter(
            "kapel_preemptions_total",
            "Requests suspended for lack of pages.",
            get(&self.preemptions),
        );
        counter(
            "kapel_prefix_cache_hit_tokens_total",
            "Prompt positions taken from shared pages rather than computed.",
            get(&self.shared_positions),
        );
        counter(
            "kapel_passes_total",
            "Passes of the model.",
            get(&self.passes),
        );
        counter(
            "kapel_recorded_passes_total",
            "Passes of the model replayed from a recording.",
            get(&self.recorded_passes),
        );
        let mut gauge = |name: &str, help: &str, value: f64| {
            let _ = writeln!(
                out,
                "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {value}"
            );
        };
        gauge(
            "kapel_requests_waiting",
            "Requests waiting to be admitted.",
            get(&self.waiting) as f64,
        );
        gauge(
            "kapel_requests_running",
            "Requests whose sequences are in the cache.",
            get(&self.running) as f64,
        );
        gauge(
            "kapel_cache_pages",
            "Pages of the cache of keys and values.",
            get(&self.pages) as f64,
        );
        let pages = get(&self.pages).max(1) as f64;
        gauge(
            "kapel_cache_usage_ratio",
            "Share of the pages of the cache in use.",
            1.0 - get(&self.free_pages) as f64 / pages,
        );
        self.first_token.render(
            &mut out,
            "kapel_time_to_first_token_seconds",
            "From the arrival of a request to its first token.",
        );
        self.between_tokens.render(
            &mut out,
            "kapel_inter_token_latency_seconds",
            "Between two tokens of an answer.",
        );
        self.duration.render(
            &mut out,
            "kapel_request_duration_seconds",
            "From the arrival of a request to its last token.",
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histograms_count_cumulated_buckets() {
        let metrics = Metrics::new();
        metrics.first_token.observe(0.004);
        metrics.first_token.observe(0.2);
        metrics.first_token.observe(100.0);
        metrics.ended(Outcome::Ok);
        let text = metrics.render();
        assert!(text.contains("kapel_time_to_first_token_seconds_bucket{le=\"0.005\"} 1"));
        assert!(text.contains("kapel_time_to_first_token_seconds_bucket{le=\"0.25\"} 2"));
        assert!(text.contains("kapel_time_to_first_token_seconds_bucket{le=\"+Inf\"} 3"));
        assert!(text.contains("kapel_time_to_first_token_seconds_count 3"));
        assert!(text.contains("kapel_requests_total{outcome=\"ok\"} 1"));
    }
}
