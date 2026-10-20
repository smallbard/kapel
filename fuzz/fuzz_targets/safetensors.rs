//! Any bytes as a safetensors file: errors are fine, panics and hangs are not.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(file) = kapel_hub::SafeTensors::from_bytes(data.to_vec()) else { return };
    let names: Vec<String> = file.names().map(str::to_string).collect();
    for name in names {
        if let Ok(tensor) = file.tensor(&name) {
            if tensor.numel() <= 1 << 16 {
                let _ = tensor.to_f64_vec();
            }
        }
    }
});
