//! Loads a model directory and writes it again with Kapel, to check that
//! `transformers` reads what Kapel writes.
//!
//! `cargo run --release -p kapel-models --example resave <from> <to>`

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, from, to] = &args[..] else {
        return Err("usage: resave <from> <to>".into());
    };
    let model = kapel_models::Model::load(&kapel_hub::Checkpoint::open(from)?)?;
    model.save(to)?;
    println!("{} written to {to}", model.config().model_type);
    Ok(())
}
