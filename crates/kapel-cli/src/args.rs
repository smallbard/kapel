//! The arguments of `kapel run` and `kapel serve`, and the options of
//! loading that `kapel eval` shares.

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

/// How to load the model: what `kapel run` and `kapel serve` share.
#[derive(Clone, Debug)]
pub(crate) struct Load {
    pub(crate) model: String,
    pub(crate) device: Device,
    pub(crate) dtype: Option<DType>,
    pub(crate) revision: String,
    pub(crate) byte_activations: bool,
    pub(crate) quantize: Option<DType>,
    pub(crate) quantize_output: bool,
    pub(crate) kv_cache: Option<DType>,
}

impl Default for Load {
    fn default() -> Self {
        Self {
            model: String::new(),
            device: Device::Cpu,
            dtype: None,
            revision: "main".into(),
            byte_activations: false,
            quantize: None,
            quantize_output: false,
            kv_cache: None,
        }
    }
}

/// `kapel run`.
#[derive(Clone, Debug)]
pub(crate) struct Run {
    pub(crate) load: Load,
    pub(crate) prompt: Option<String>,
    pub(crate) raw: bool,
    pub(crate) system: Option<String>,
    pub(crate) max_tokens: usize,
    pub(crate) sampling: SamplingOptions,
    pub(crate) stop: Vec<String>,
    pub(crate) stats: bool,
    /// Speculative decoding: a smaller model proposes.
    pub(crate) draft: Option<String>,
    /// Or the tokens of the sequence (prompt lookup).
    pub(crate) prompt_lookup: bool,
    /// Or the model's multi-token layer.
    pub(crate) multi_token: bool,
    /// The proposals per pass.
    pub(crate) speculate: usize,
}

/// `kapel serve`.
#[derive(Clone, Debug)]
pub(crate) struct Serve {
    pub(crate) load: Load,
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) name: Option<String>,
    pub(crate) max_model_len: Option<usize>,
    /// The memory of the cache, in GiB; else from the card's memory.
    pub(crate) cache_memory: Option<f64>,
    pub(crate) gpu_memory_utilization: f64,
    pub(crate) max_sequences: usize,
    pub(crate) max_pass_tokens: usize,
    pub(crate) max_requests: usize,
    pub(crate) prefix_sharing: bool,
    pub(crate) timeout: u64,
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

pub(crate) fn number<T: std::str::FromStr>(option: &str, value: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("{option}: {value:?} is not a valid number"))
}

/// Reads an option of [`Load`]: whether it was one.
fn load_option(
    load: &mut Load,
    option: &str,
    value: &mut dyn FnMut() -> Result<String, String>,
) -> Result<bool, String> {
    match option {
        "--device" => {
            let v = value()?;
            load.device = match v.as_str() {
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
            load.dtype = Some(match v.as_str() {
                "f32" => DType::F32,
                "bf16" => DType::BF16,
                "f16" => DType::F16,
                _ => return Err(format!("--dtype: {v:?} is not f32, bf16 or f16")),
            })
        }
        "--revision" => load.revision = value()?,
        "--byte-activations" => load.byte_activations = true,
        "--quantize" => load.quantize = Some(format(&value()?)?),
        "--quantize-output" => load.quantize_output = true,
        "--kv-cache" => {
            let v = value()?;
            load.kv_cache = Some(match v.as_str() {
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
        _ => return Ok(false),
    }
    Ok(true)
}

/// Walks `args`: each option with `option` (which reads its value with the
/// closure it is given), the model's name aside.
pub(crate) fn walk(
    command: &str,
    args: &[String],
    load: &mut Load,
    mut option: impl FnMut(&str, &mut dyn FnMut() -> Result<String, String>) -> Result<bool, String>,
) -> Result<(), String> {
    let mut model = None;
    let mut i = 0;
    while i < args.len() {
        let name = args[i].clone();
        let mut value = || {
            i += 1;
            args.get(i)
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        let known = load_option(load, &name, &mut value)? || option(&name, &mut value)?;
        if known {
            // Read.
        } else if name.starts_with("--") {
            return Err(format!("unknown option {name}"));
        } else if model.is_none() {
            model = Some(name);
        } else {
            return Err(format!("unexpected argument {name:?}"));
        }
        i += 1;
    }
    load.model = model.ok_or(format!(
        "{command}: which model? (a directory or owner/name)"
    ))?;
    Ok(())
}

/// Reads the arguments that follow `run`.
pub(crate) fn parse(args: &[String]) -> Result<Run, String> {
    let mut run = Run {
        load: Load::default(),
        prompt: None,
        raw: false,
        system: None,
        max_tokens: 512,
        sampling: SamplingOptions::default(),
        stop: Vec::new(),
        stats: false,
        draft: None,
        prompt_lookup: false,
        multi_token: false,
        speculate: 4,
    };
    let mut load = Load::default();
    walk("run", args, &mut load, |option, value| {
        match option {
            "--prompt" => run.prompt = Some(value()?),
            "--raw" => run.raw = true,
            "--system" => run.system = Some(value()?),
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
            "--stats" => run.stats = true,
            "--draft" => run.draft = Some(value()?),
            "--prompt-lookup" => run.prompt_lookup = true,
            "--multi-token" => run.multi_token = true,
            "--speculate" => run.speculate = number(option, &value()?)?,
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    run.load = load;
    if usize::from(run.draft.is_some())
        + usize::from(run.prompt_lookup)
        + usize::from(run.multi_token)
        > 1
    {
        return Err("one of --draft, --prompt-lookup and --multi-token".into());
    }
    Ok(run)
}

/// Reads the arguments that follow `serve`.
pub(crate) fn parse_serve(args: &[String]) -> Result<Serve, String> {
    let mut serve = Serve {
        load: Load::default(),
        host: "127.0.0.1".into(),
        port: 8000,
        name: None,
        max_model_len: None,
        cache_memory: None,
        gpu_memory_utilization: 0.85,
        max_sequences: 256,
        max_pass_tokens: 2048,
        max_requests: 1024,
        prefix_sharing: true,
        timeout: 600,
    };
    let mut load = Load::default();
    walk("serve", args, &mut load, |option, value| {
        match option {
            "--host" => serve.host = value()?,
            "--port" => serve.port = number(option, &value()?)?,
            "--served-model-name" => serve.name = Some(value()?),
            "--max-model-len" => serve.max_model_len = Some(number(option, &value()?)?),
            "--cache-memory" => serve.cache_memory = Some(number(option, &value()?)?),
            "--gpu-memory-utilization" => {
                serve.gpu_memory_utilization = number(option, &value()?)?;
                if !(0.05..=1.0).contains(&serve.gpu_memory_utilization) {
                    return Err(format!("{option}: between 0.05 and 1"));
                }
            }
            "--max-sequences" => serve.max_sequences = number(option, &value()?)?,
            "--max-pass-tokens" => serve.max_pass_tokens = number(option, &value()?)?,
            "--max-requests" => serve.max_requests = number(option, &value()?)?,
            "--no-prefix-sharing" => serve.prefix_sharing = false,
            "--timeout" => serve.timeout = number(option, &value()?)?,
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    serve.load = load;
    Ok(serve)
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
        assert_eq!(run.load.model, "m");
        assert_eq!(run.load.device, Device::Cuda(1));
        assert_eq!(run.load.dtype, Some(DType::BF16));
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
        assert_eq!(run.load.quantize, Some(DType::int4(32).unwrap()));
        assert!(run.load.quantize_output);
        assert_eq!(
            parse_all("m --quantize int8").unwrap().load.quantize,
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

    #[test]
    fn speculation_is_read() {
        let run = parse_all("m --draft small --speculate 6").unwrap();
        assert_eq!(run.draft.as_deref(), Some("small"));
        assert_eq!(run.speculate, 6);
        assert!(parse_all("m --draft small --multi-token").is_err());
        assert!(parse_all("m --prompt-lookup").unwrap().prompt_lookup);
    }

    #[test]
    fn server_options_are_read() {
        let args: Vec<String> =
            "m --device cuda --port 9000 --kv-cache f8e4m3 --max-model-len 4096 \
                                 --no-prefix-sharing --served-model-name llama"
                .split_whitespace()
                .map(str::to_string)
                .collect();
        let serve = parse_serve(&args).unwrap();
        assert_eq!(serve.load.model, "m");
        assert_eq!(serve.load.device, Device::Cuda(0));
        assert_eq!(serve.load.kv_cache, Some(DType::F8E4M3));
        assert_eq!(serve.port, 9000);
        assert_eq!(serve.max_model_len, Some(4096));
        assert!(!serve.prefix_sharing);
        assert_eq!(serve.name.as_deref(), Some("llama"));
        let bad: Vec<String> = vec!["m".into(), "--prompt".into(), "x".into()];
        assert!(parse_serve(&bad).unwrap_err().contains("unknown option"));
        let none: Vec<String> = vec!["--port".into(), "1".into()];
        assert!(
            parse_serve(&none)
                .unwrap_err()
                .contains("serve: which model")
        );
    }
}
