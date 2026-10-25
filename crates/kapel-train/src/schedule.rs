//! Learning rates along a run.

use crate::config::ScheduleConfig;

/// The learning rate of each step, as a factor of the peak.
#[derive(Clone, Debug, PartialEq)]
pub struct Schedule {
    kind: Kind,
    warmup: usize,
    steps: usize,
    last: f64,
    decay: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Constant,
    Cosine,
    Wsd,
    Linear,
}

impl Schedule {
    /// The schedule a checked configuration describes.
    pub fn new(config: &ScheduleConfig) -> Self {
        Self {
            kind: match config.kind.as_str() {
                "constant" => Kind::Constant,
                "wsd" => Kind::Wsd,
                "linear" => Kind::Linear,
                _ => Kind::Cosine,
            },
            warmup: config.warmup,
            steps: config.steps,
            last: config.r#final,
            decay: config.decay,
        }
    }

    /// The factor of step `step`, counted from 0:
    ///
    /// - during the `warmup` first steps, `(step + 1) / warmup`, as llm.c;
    /// - then, for `"cosine"`, from 1 to `final` along half a cosine, as
    ///   llm.c and nanoGPT; for `"wsd"`, 1, then a straight line from 1 to
    ///   `final` over the last `decay` of the run; for `"linear"`, from 1 at
    ///   the end of the warmup to `final` one step past the last, as
    ///   `transformers`' `get_linear_schedule_with_warmup` (TRL's default):
    ///   the last step has `1 / (steps − warmup)` of the way left; for
    ///   `"constant"`, 1.
    pub fn factor(&self, step: usize) -> f64 {
        if step < self.warmup {
            return (step + 1) as f64 / self.warmup as f64;
        }
        let last = self.last;
        match self.kind {
            Kind::Constant => 1.0,
            Kind::Linear => {
                let span = (self.steps - self.warmup).max(1) as f64;
                let left = ((self.steps - step.min(self.steps)) as f64 / span).max(0.0);
                last + left * (1.0 - last)
            }
            Kind::Cosine => {
                let span = (self.steps - self.warmup).max(1) as f64;
                let ratio = ((step - self.warmup) as f64 / span).min(1.0);
                let coefficient = 0.5 * (1.0 + (std::f64::consts::PI * ratio).cos());
                last + coefficient * (1.0 - last)
            }
            Kind::Wsd => {
                let decaying = (self.decay * self.steps as f64).round() as usize;
                let start = self.steps - decaying.min(self.steps);
                if step < start || decaying == 0 {
                    1.0
                } else {
                    let ratio = ((step - start) as f64 / decaying as f64).min(1.0);
                    1.0 - ratio * (1.0 - last)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule(kind: &str) -> Schedule {
        Schedule::new(&ScheduleConfig {
            kind: kind.into(),
            warmup: 10,
            steps: 110,
            r#final: 0.1,
            decay: 0.5,
        })
    }

    #[test]
    fn warmup_then_the_shape_of_each_kind() {
        let c = schedule("cosine");
        assert_eq!(c.factor(0), 0.1);
        assert_eq!(c.factor(9), 1.0);
        assert_eq!(c.factor(10), 1.0);
        assert!((c.factor(60) - 0.55).abs() < 1e-12);
        assert!((c.factor(110) - 0.1).abs() < 1e-12);
        let w = schedule("wsd");
        assert_eq!(w.factor(54), 1.0);
        assert_eq!(w.factor(55), 1.0);
        assert!((w.factor(110) - 0.1).abs() < 1e-12);
        assert!(w.factor(80) < 1.0 && w.factor(80) > 0.1);
        assert_eq!(schedule("constant").factor(100), 1.0);
        let l = schedule("linear");
        assert_eq!(l.factor(10), 1.0);
        assert!((l.factor(60) - 0.55).abs() < 1e-12);
        assert!((l.factor(109) - (0.1 + 0.9 / 100.0)).abs() < 1e-12);
    }
}
