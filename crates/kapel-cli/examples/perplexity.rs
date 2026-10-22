//! The perplexity of a model on a text, as llama.cpp's `llama-perplexity`
//! computes it: the tokens of the text cut into chunks of `--ctx` tokens, the
//! first of each replaced by the beginning of sequence when the tokenizer adds
//! one, and each chunk scored on its second half. The chunks are computed by
//! pieces of `--batch` tokens after the cache, with the kernels of generation.
//!
//! `cargo run --release -p kapel-cli --example perplexity -- MODEL TEXT
//! [--ctx 512] [--chunks N] [--batch 16] [--byte-activations]
//! [--quantize int8|int4|q8_0|q4_0] [--quantize-output]`
//!
//! `TEXT` is a text file, such as `wiki.test.raw` of wikitext-2 (see
//! `tools/parity/README.md`).

use std::time::Instant;

use kapel_core::DType;
use kapel_infer::Pipeline;
use kapel_models::Cache;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: perplexity MODEL TEXT [--ctx 512] [--chunks N] [--batch 16] \
                 [--byte-activations] [--quantize int8|int4|q8_0|q4_0] [--quantize-output]";
    let mut options = kapel_infer::LoadOptions {
        dtype: Some(DType::F32),
        ..Default::default()
    };
    let (mut ctx, mut chunks, mut batch, mut bytes) = (512usize, usize::MAX, 16usize, false);
    let mut positional = Vec::new();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--quantize" {
            options.quantize = Some(match args.next().as_deref() {
                Some("int8") => DType::int8(128)?,
                Some("int4") => DType::int4(64)?,
                Some("int4/32") => DType::int4(32)?,
                Some("q8_0") => DType::Quantized(kapel_core::QuantFormat::Q8_0),
                Some("q4_0") => DType::Quantized(kapel_core::QuantFormat::Q4_0),
                _ => return Err(usage.into()),
            });
            continue;
        }
        if arg == "--quantize-output" {
            options.quantize_output = true;
            continue;
        }
        let mut value =
            || -> Result<usize, &str> { args.next().and_then(|v| v.parse().ok()).ok_or(usage) };
        match arg.as_str() {
            "--ctx" => ctx = value()?,
            "--chunks" => chunks = value()?,
            "--batch" => batch = value()?.max(1),
            "--byte-activations" => bytes = true,
            _ => positional.push(arg),
        }
    }
    let [model, text] = positional.as_slice() else {
        return Err(usage.into());
    };
    let cpu = kapel_cpu::Cpu::new().with_byte_activations(bytes);
    let pipeline = Pipeline::load_with(model, cpu.clone(), &options)?;
    let tokenizer = pipeline.tokenizer();
    let tokens = tokenizer.encode(&std::fs::read_to_string(text)?, true)?;
    // The beginning of sequence, if the tokenizer adds one.
    let bos = tokenizer.encode("", true)?.first().copied();
    let count = chunks.min(tokens.len() / ctx);
    println!(
        "{} tokens, {count} chunks of {ctx}, activations in bytes: {bytes}, quantized: {:?}",
        tokens.len(),
        options.quantize.map(|q| q.to_string())
    );
    let first = ctx / 2;
    let (mut nll, mut scored) = (0.0f64, 0usize);
    let start = Instant::now();
    for chunk in 0..count {
        let mut ids = tokens[chunk * ctx..(chunk + 1) * ctx].to_vec();
        if let Some(bos) = bos {
            ids[0] = bos;
        }
        let mut cache = Cache::new();
        for (piece, part) in ids.chunks(batch).enumerate() {
            let logits = pipeline.model().extend(&cpu, part, &mut cache)?;
            let values = logits.to_f64_vec()?;
            let vocab = values.len() / part.len();
            for (j, row) in values.chunks_exact(vocab).enumerate() {
                let pos = piece * batch + j;
                if pos < first || pos + 1 >= ctx {
                    continue;
                }
                let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = row.iter().map(|v| (v - max).exp()).sum();
                nll += max + sum.ln() - row[ids[pos + 1] as usize];
                scored += 1;
            }
        }
        print!("[{}]{:.4},", chunk + 1, (nll / scored as f64).exp());
        use std::io::Write;
        std::io::stdout().flush()?;
    }
    println!();
    println!(
        "perplexity {:.4} over {scored} tokens, in {:.0} s",
        (nll / scored as f64).exp(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
