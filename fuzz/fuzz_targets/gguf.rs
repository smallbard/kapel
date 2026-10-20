//! Any bytes as a GGUF file, its tensors dequantized.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(file) = kapel_hub::Gguf::from_bytes(data.to_vec()) else { return };
    let names: Vec<String> = file.names().map(str::to_string).collect();
    for name in names {
        if let Ok(tensor) = file.tensor(&name) {
            if tensor.numel() <= 1 << 16 {
                let _ = tensor.to_f64_vec();
            }
        }
    }
});
