//! `kapel quantize`: a model quantized and written, to GGUF or to a model
//! directory.

use std::path::{Path, PathBuf};
use std::time::Instant;

use kapel_core::{DType, QuantFormat};
use kapel_infer::{LoadOptions, Pipeline};
use kapel_models::Model;
use kapel_ops::{Backend, Run, Transfer};
use kapel_quant::{Method, Options};

use crate::args::Device;

pub(crate) const USAGE: &str = "\
usage: kapel quantize MODEL --output PATH [options]

Quantizes the linear projections of MODEL (a model directory, a GGUF file or a
repository of the Hub) and writes the model to PATH: a GGUF file if PATH ends
with .gguf, which llama.cpp reads too, else a model directory.

options:
  --format FORMAT          q4_0, q4_1, q8_0 (GGUF), int4 and int8 (groups of 64
                           and 128, or int4/G, int8/G), or f16, bf16, f32 to
                           convert without quantizing (default q4_0)
  --method METHOD          rtn: rounding to the nearest, without data (default);
                           gptq; awq; awq+gptq
  --calibration FILE       the text GPTQ and AWQ calibrate on
  --samples N              windows of the text, drawn at random (default 128)
  --length N               tokens per window (default 512)
  --seed S                 the draw of the windows (default 0)
  --quantize-output        quantize the output table too, by rounding
  --device cpu|cuda[:N]    where GPTQ and AWQ compute (default cpu)
  --revision REV           the revision of a Hub repository (default main)
";

/// What `kapel quantize` was asked.
#[derive(Clone, Debug)]
struct Quantize {
    model: String,
    output: PathBuf,
    format: DType,
    method: Method,
    calibration: Option<PathBuf>,
    samples: usize,
    length: usize,
    seed: u64,
    quantize_output: bool,
    device: Device,
    revision: String,
}

/// A format of `--format`.
fn format(value: &str) -> Result<DType, String> {
    let bad = || format!("--format: {value:?} is not a format of kapel quantize");
    let (name, group) = match value.split_once('/') {
        Some((name, group)) => (name, Some(group.parse::<u16>().map_err(|_| bad())?)),
        None => (value, None),
    };
    Ok(match (name, group) {
        ("int8", g) => DType::int8(g.unwrap_or(128)).map_err(|e| e.to_string())?,
        ("int4", g) => DType::int4(g.unwrap_or(64)).map_err(|e| e.to_string())?,
        ("q8_0", None) => DType::Quantized(QuantFormat::Q8_0),
        ("q4_0", None) => DType::Quantized(QuantFormat::Q4_0),
        ("q4_1", None) => DType::Quantized(QuantFormat::Q4_1),
        ("f16", None) => DType::F16,
        ("bf16", None) => DType::BF16,
        ("f32", None) => DType::F32,
        _ => return Err(bad()),
    })
}

fn parse(args: &[String]) -> Result<Quantize, String> {
    let mut q = Quantize {
        model: String::new(),
        output: PathBuf::new(),
        format: DType::Quantized(QuantFormat::Q4_0),
        method: Method::Rtn,
        calibration: None,
        samples: 128,
        length: 512,
        seed: 0,
        quantize_output: false,
        device: Device::Cpu,
        revision: "main".into(),
    };
    let mut model = None;
    let mut output = None;
    let mut i = 0;
    while i < args.len() {
        let name = args[i].as_str();
        let mut value = || {
            i += 1;
            args.get(i)
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        let number = |v: String| {
            v.parse::<usize>()
                .map_err(|_| format!("{name}: {v:?} is not a number"))
        };
        match name {
            "--output" | "-o" => output = Some(PathBuf::from(value()?)),
            "--format" => q.format = format(&value()?)?,
            "--method" => {
                let v = value()?;
                q.method = Method::from_name(&v)
                    .ok_or_else(|| format!("--method: {v:?} is not rtn, gptq, awq or awq+gptq"))?;
            }
            "--calibration" => q.calibration = Some(PathBuf::from(value()?)),
            "--samples" => q.samples = number(value()?)?,
            "--length" => q.length = number(value()?)?.max(2),
            "--seed" => q.seed = number(value()?)? as u64,
            "--quantize-output" => q.quantize_output = true,
            "--device" => {
                let v = value()?;
                q.device = match v.as_str() {
                    "cpu" => Device::Cpu,
                    "cuda" => Device::Cuda(0),
                    other => match other.strip_prefix("cuda:").and_then(|n| n.parse().ok()) {
                        Some(n) => Device::Cuda(n),
                        None => return Err(format!("--device: {v:?} is not cpu or cuda[:N]")),
                    },
                }
            }
            "--revision" => q.revision = value()?,
            other if other.starts_with('-') => return Err(format!("unknown option {other}")),
            other if model.is_none() => model = Some(other.to_string()),
            other => return Err(format!("unexpected argument {other:?}")),
        }
        i += 1;
    }
    q.model = model.ok_or("quantize: which model? (a directory, a GGUF file or owner/name)")?;
    q.output = output.ok_or("quantize: where to? (--output PATH)")?;
    if q.method.calibrates() && q.calibration.is_none() {
        return Err(format!(
            "--method {} needs a text to calibrate on (--calibration FILE)",
            q.method.name()
        ));
    }
    if q.format.is_float() && q.method != Method::Rtn {
        return Err(format!(
            "--format {} converts; it takes no --method",
            q.format
        ));
    }
    Ok(q)
}

/// `samples` windows of `length` tokens of `tokens`, drawn at random with
/// `seed` (the same windows for the same seed), each beginning with `bos`
/// when there is one.
fn windows(
    tokens: &[u32],
    samples: usize,
    length: usize,
    seed: u64,
    bos: Option<u32>,
) -> Vec<Vec<u32>> {
    let body = length - usize::from(bos.is_some());
    if tokens.len() <= body {
        return vec![bos.into_iter().chain(tokens.iter().copied()).collect()];
    }
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..samples)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let start = (state % (tokens.len() - body) as u64) as usize;
            bos.into_iter()
                .chain(tokens[start..start + body].iter().copied())
                .collect()
        })
        .collect()
}

/// The files of a model directory that are not its weights, copied to a
/// directory written: the tokenizer, the chat template, the defaults of
/// generation.
const COMPANIONS: [&str; 6] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
    "chat_template.json",
];

pub(crate) fn execute(args: &[String]) -> Result<(), String> {
    let q = parse(args)?;
    let source = crate::directory(&q.model, &q.revision)?;
    let cpu = kapel_cpu::Cpu::new();
    // The model as stored, in host memory, and its tokenizer.
    let options = LoadOptions {
        dtype: None,
        quantize: None,
        quantize_output: false,
        fuse: false,
        cache_dtype: None,
    };
    let pipeline =
        Pipeline::load_with(&source, cpu.clone(), &options).map_err(|e| e.to_string())?;
    let calibration = match &q.calibration {
        Some(path) => {
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let tokenizer = pipeline.tokenizer();
            let tokens = tokenizer.encode(&text, false).map_err(|e| e.to_string())?;
            let bos = tokenizer
                .encode("", true)
                .map_err(|e| e.to_string())?
                .first()
                .copied();
            windows(&tokens, q.samples, q.length, q.seed, bos)
        }
        None => Vec::new(),
    };
    let start = Instant::now();
    let model = pipeline.model();
    let quantized = if q.format.is_float() {
        // The matrices; the norms and biases keep their type (llama.cpp's
        // converter keeps them in `f32`).
        let mut converted = model.clone();
        for name in model.names() {
            let w = model.tensor(name).expect("listed");
            if w.rank() == 2 && w.dtype().is_float() && w.dtype() != q.format {
                let cast = cpu
                    .forward(&kapel_ops::Cast { dtype: q.format }, &[w])
                    .map_err(|e| e.to_string())?;
                converted.replace(name, cast);
            }
        }
        converted
    } else {
        let mut options = Options::new(q.format, q.method);
        options.output = q.quantize_output;
        match q.device {
            Device::Cpu => run(model, &cpu, DType::F32, &calibration, &options)?,
            #[cfg(feature = "cuda")]
            Device::Cuda(ordinal) => {
                let cuda = kapel_cuda::Cuda::new(ordinal).map_err(|e| e.to_string())?;
                run(model, &cuda, DType::BF16, &calibration, &options)?
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => return Err("this kapel was built without the feature `cuda`".into()),
        }
    };
    eprintln!(
        "kapel: quantized in {:.0} s; writing {}",
        start.elapsed().as_secs_f64(),
        q.output.display()
    );
    write(
        &quantized,
        &cpu,
        &source,
        &crate::model_name(&q.model),
        &q.output,
    )
}

fn run<B: Backend + Transfer>(
    model: &Model,
    backend: &B,
    compute: DType,
    calibration: &[Vec<u32>],
    options: &Options,
) -> Result<Model, String> {
    kapel_quant::quantize(model, backend, compute, calibration, options, &mut |line| {
        eprintln!("kapel: {line}");
    })
    .map_err(|e| e.to_string())
}

/// Writes `model` to `output`: GGUF for a `.gguf` path, else a model
/// directory with the tokenizer's files of `source`.
fn write(
    model: &Model,
    cpu: &kapel_cpu::Cpu,
    source: &Path,
    name: &str,
    output: &Path,
) -> Result<(), String> {
    if output.extension().is_some_and(|e| e == "gguf") {
        return kapel_infer::save_gguf(model, cpu, source, Some(name), output)
            .map_err(|e| e.to_string());
    }
    if !source.is_dir() {
        return Err("a model read from GGUF is written to GGUF (--output NAME.gguf)".into());
    }
    model.save(output).map_err(|e| e.to_string())?;
    for name in COMPANIONS {
        let from = source.join(name);
        if from.exists() {
            std::fs::copy(&from, output.join(name))
                .map_err(|e| format!("{}: {e}", from.display()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_all(line: &str) -> Result<Quantize, String> {
        let args: Vec<String> = line.split(' ').map(str::to_string).collect();
        parse(&args)
    }

    #[test]
    fn options_are_read() {
        let q =
            parse_all("m -o out.gguf --format q4_1 --method gptq --calibration t.txt --samples 8")
                .unwrap();
        assert_eq!(q.model, "m");
        assert_eq!(q.format, DType::Quantized(QuantFormat::Q4_1));
        assert_eq!(q.method, Method::Gptq);
        assert_eq!(q.samples, 8);
        assert_eq!(
            parse_all("m -o d --format int4/32").unwrap().format,
            DType::int4(32).unwrap()
        );
        assert!(
            parse_all("m -o d --method gptq")
                .unwrap_err()
                .contains("--calibration")
        );
        assert!(
            parse_all("m --format q4_0")
                .unwrap_err()
                .contains("--output")
        );
        assert!(parse_all("m -o d --format q3").is_err());
        assert!(parse_all("m -o d --format f16 --method awq --calibration t").is_err());
    }

    #[test]
    fn windows_are_drawn_again_alike() {
        let tokens: Vec<u32> = (0..1000).collect();
        let a = windows(&tokens, 5, 16, 3, Some(7));
        assert_eq!(a, windows(&tokens, 5, 16, 3, Some(7)));
        assert_ne!(a, windows(&tokens, 5, 16, 4, Some(7)));
        assert!(a.iter().all(|w| w.len() == 16 && w[0] == 7));
        assert!(a.iter().all(|w| w[2] == w[1] + 1));
        // A text shorter than a window is one window.
        assert_eq!(
            windows(&tokens[..5], 5, 16, 0, None),
            [tokens[..5].to_vec()]
        );
    }
}
