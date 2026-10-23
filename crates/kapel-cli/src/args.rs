//! The arguments of `kapel run`.

use kapel_core::DType;
use kapel_infer::Sampling;

/// Where to compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Device {
    Cpu,
    Cuda(u32),
}

/// The sampling options given, over the model's defaults.
#[derive(Clone, Debug, Default)]
pub(crate) struct SamplingOptions {
    greedy: bool,
    temperature: Option<f64>,
    top_k: Option<usize>,
    top_p: Option<f64>,
    min_p: Option<f64>,
    repetition_penalty: Option<f64>,
    seed: Option<u64>,
}

impl SamplingOptions {
    /// `sampling` with the options given.
    pub(crate) fn apply(&self, sampling: &mut Sampling) {
        if let Some(t) = self.temperature {
            sampling.temperature = t;
        }
        if self.greedy {
            sampling.temperature = 0.0;
        }
        if let Some(k) = self.top_k {
            sampling.top_k = (k > 0).then_some(k);
        }
        if let Some(p) = self.top_p {
            sampling.top_p = p;
        }
        if let Some(p) = self.min_p {
            sampling.min_p = p;
        }
        if let Some(r) = self.repetition_penalty {
            sampling.repetition_penalty = r;
        }
        if let Some(s) = self.seed {
            sampling.seed = s;
        }
    }
}

/// `kapel run`.
#[derive(Clone, Debug)]
pub(crate) struct Run {
    pub(crate) model: String,
    pub(crate) prompt: Option<String>,
    pub(crate) raw: bool,
    pub(crate) system: Option<String>,
    pub(crate) device: Device,
    pub(crate) dtype: Option<DType>,
    pub(crate) max_tokens: usize,
    pub(crate) sampling: SamplingOptions,
    pub(crate) stop: Vec<String>,
    pub(crate) revision: String,
    pub(crate) stats: bool,
    pub(crate) byte_activations: bool,
    pub(crate) quantize: Option<DType>,
    pub(crate) quantize_output: bool,
    pub(crate) kv_cache: Option<DType>,
}

/// A quantized format: `int8` and `int4` (groups of 128 and 64, or `/G`),
/// `q8_0`, `q4_0`.
fn format(value: &str) -> Result<DType, String> {
    let bad = || format!("--quantize: {value:?} is not int8[/G], int4[/G], q8_0 or q4_0");
    let (name, group) = match value.split_once('/') {
        Some((name, group)) => (name, Some(group.parse::<u16>().map_err(|_| bad())?)),
        None => (value, None),
    };
    match name {
        "int8" => DType::int8(group.unwrap_or(128)).map_err(|e| e.to_string()),
        "int4" => DType::int4(group.unwrap_or(64)).map_err(|e| e.to_string()),
        "q8_0" if group.is_none() => Ok(DType::Quantized(kapel_core::QuantFormat::Q8_0)),
        "q4_0" if group.is_none() => Ok(DType::Quantized(kapel_core::QuantFormat::Q4_0)),
        _ => Err(bad()),
    }
}

fn number<T: std::str::FromStr>(option: &str, value: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("{option}: {value:?} is not a valid number"))
}

/// Reads the arguments that follow `run`.
pub(crate) fn parse(args: &[String]) -> Result<Run, String> {
    let mut run = Run {
        model: String::new(),
        prompt: None,
        raw: false,
        system: None,
        device: Device::Cpu,
        dtype: None,
        max_tokens: 512,
        sampling: SamplingOptions::default(),
        stop: Vec::new(),
        revision: "main".into(),
        stats: false,
        byte_activations: false,
        quantize: None,
        quantize_output: false,
        kv_cache: None,
    };
    let mut model = None;
    let mut i = 0;
    while i < args.len() {
        let option = args[i].as_str();
        let mut value = || {
            i += 1;
            args.get(i)
                .cloned()
                .ok_or_else(|| format!("{option} needs a value"))
        };
        match option {
            "--prompt" => run.prompt = Some(value()?),
            "--raw" => run.raw = true,
            "--system" => run.system = Some(value()?),
            "--device" => {
                let v = value()?;
                run.device = match v.as_str() {
                    "cpu" => Device::Cpu,
                    "cuda" => Device::Cuda(0),
                    other => match other.strip_prefix("cuda:") {
                        Some(n) => Device::Cuda(number("--device", n)?),
                        None => return Err(format!("--device: {v:?} is not cpu or cuda[:N]")),
                    },
                }
            }
            "--dtype" => {
                let v = value()?;
                run.dtype = Some(match v.as_str() {
                    "f32" => DType::F32,
                    "bf16" => DType::BF16,
                    "f16" => DType::F16,
                    _ => return Err(format!("--dtype: {v:?} is not f32, bf16 or f16")),
                })
            }
            "--max-tokens" => run.max_tokens = number(option, &value()?)?,
            "--greedy" => run.sampling.greedy = true,
            "--temperature" => run.sampling.temperature = Some(number(option, &value()?)?),
            "--top-k" => run.sampling.top_k = Some(number(option, &value()?)?),
            "--top-p" => run.sampling.top_p = Some(number(option, &value()?)?),
            "--min-p" => run.sampling.min_p = Some(number(option, &value()?)?),
            "--repetition-penalty" => {
                run.sampling.repetition_penalty = Some(number(option, &value()?)?)
            }
            "--seed" => run.sampling.seed = Some(number(option, &value()?)?),
            "--stop" => run.stop.push(value()?),
            "--revision" => run.revision = value()?,
            "--stats" => run.stats = true,
            "--byte-activations" => run.byte_activations = true,
            "--quantize" => run.quantize = Some(format(&value()?)?),
            "--quantize-output" => run.quantize_output = true,
            "--kv-cache" => {
                let v = value()?;
                run.kv_cache = Some(match v.as_str() {
                    "f32" => DType::F32,
                    "bf16" => DType::BF16,
                    "f16" => DType::F16,
                    "f8e4m3" => DType::F8E4M3,
                    "f8e5m2" => DType::F8E5M2,
                    _ => {
                        return Err(format!(
                            "--kv-cache: {v:?} is not f32, bf16, f16, f8e4m3 or f8e5m2"
                        ));
                    }
                })
            }
            other if other.starts_with("--") => return Err(format!("unknown option {other}")),
            other if model.is_none() => model = Some(other.to_string()),
            other => return Err(format!("unexpected argument {other:?}")),
        }
        i += 1;
    }
    run.model = model.ok_or("run: which model? (a directory or owner/name)")?;
    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_all(line: &str) -> Result<Run, String> {
        let args: Vec<String> = line.split(' ').map(str::to_string).collect();
        parse(&args)
    }

    #[test]
    fn options_are_read() {
        let run =
            parse_all("m --device cuda:1 --dtype bf16 --top-k 0 --stop END --stop STOP --greedy")
                .unwrap();
        assert_eq!(run.model, "m");
        assert_eq!(run.device, Device::Cuda(1));
        assert_eq!(run.dtype, Some(DType::BF16));
        assert_eq!(run.stop, ["END", "STOP"]);
        let mut sampling = Sampling {
            temperature: 0.7,
            top_k: Some(20),
            ..Sampling::default()
        };
        run.sampling.apply(&mut sampling);
        assert!(sampling.is_greedy());
        assert_eq!(sampling.top_k, None);
    }

    #[test]
    fn mistakes_are_explained() {
        assert!(parse_all("--greedy").unwrap_err().contains("which model"));
        let run = parse_all("m --quantize int4/32 --quantize-output").unwrap();
        assert_eq!(run.quantize, Some(DType::int4(32).unwrap()));
        assert!(run.quantize_output);
        assert_eq!(
            parse_all("m --quantize int8").unwrap().quantize,
            Some(DType::int8(128).unwrap())
        );
        assert!(parse_all("m --quantize int3").is_err());
        assert!(parse_all("m --quantize q8_0/16").is_err());
        assert!(parse_all("m --top-p x").unwrap_err().contains("--top-p"));
        assert!(
            parse_all("m --device tpu")
                .unwrap_err()
                .contains("cpu or cuda")
        );
        assert!(
            parse_all("m --max-tokens")
                .unwrap_err()
                .contains("needs a value")
        );
        assert!(parse_all("m n").unwrap_err().contains("unexpected"));
    }
}
