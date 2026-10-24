//! How fast batches are read: `cargo run --release -p kapel-train --example
//! loader -- SOURCE [ROWS LENGTH]`, SOURCE a dataset of `kapel prepare` or a
//! pattern of llm.c files. Reads 200 batches as a stream, then 200 shuffled
//! whole documents with their mask, and prints the tokens per second of
//! the thread that reads them.

use std::time::Instant;

use kapel_train::data::{Loader, Pack, Reading, Source};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: loader SOURCE [ROWS LENGTH]")?;
    let rows: usize = args.next().map_or(Ok(16), |a| a.parse())?;
    let length: usize = args.next().map_or(Ok(1024), |a| a.parse())?;
    for (name, pack, mask, shuffle) in [
        ("stream, in order", Pack::Stream, false, false),
        ("documents, mask, shuffled", Pack::Documents, true, true),
    ] {
        let mut reading = Reading::new(rows, length);
        reading.pack = pack;
        reading.mask = mask;
        reading.shuffle = shuffle;
        reading.separator = Some(50256);
        let source = Source::open(&path, "train", &reading.cache)?;
        let mut loader = Loader::new(vec![(source, 1.0)], reading)?;
        // The first batch opens a shard: not counted.
        loader.next_batch()?;
        let batches = 200;
        let start = Instant::now();
        let mut real = 0usize;
        for _ in 0..batches {
            let b = loader.next_batch()?;
            real += b
                .targets
                .iter()
                .filter(|&&t| t != kapel_train::IGNORE)
                .count();
        }
        let seconds = start.elapsed().as_secs_f64();
        let tokens = batches * rows * length;
        println!(
            "{name}: {:.1} M tokens/s ({:.2} ms a batch of {rows} × {length}), {:.3} % of the targets real",
            tokens as f64 / seconds / 1e6,
            seconds * 1e3 / batches as f64,
            100.0 * real as f64 / tokens as f64
        );
    }
    Ok(())
}
