//! Recorded passes on the card (CUDA graphs): a model's next tokens computed
//! through replays of a recorded step give the logits of computing each step
//! as it comes, bit for bit, for one sequence and for several. Uses the small
//! models of `data/parity`; without them, the test is skipped.

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use kapel_core::DType;
use kapel_cuda::Cuda;
use kapel_hub::Checkpoint;
use kapel_models::{Cache, KvCache, Model};
use kapel_ops::Transfer;

/// Held by each test: a recording captures all the work of the device's
/// stream, that of the other tests too.
static ALONE: Mutex<()> = Mutex::new(());

fn alone() -> MutexGuard<'static, ()> {
    ALONE.lock().unwrap_or_else(|e| e.into_inner())
}

fn model(name: &str, cuda: &Cuda) -> Option<Model> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/parity")
        .join(name);
    let checkpoint = Checkpoint::open(&dir).ok()?;
    let model = Model::load(&checkpoint).ok()?.upload(cuda).ok()?;
    Some(model.cast(cuda, DType::BF16).unwrap().fused(cuda).unwrap())
}

fn values(cuda: &Cuda, logits: &kapel_core::Tensor) -> Vec<f64> {
    cuda.download(logits).unwrap().to_f64_vec().unwrap()
}

#[test]
fn replays_give_the_logits_of_steps_as_they_come() {
    let _alone = alone();
    let cuda = Cuda::new(0).unwrap();
    for name in ["smollm2", "llama", "qwen3"] {
        let Some(model) = model(name, &cuda) else {
            eprintln!("skipped {name}: data/parity/{name} missing");
            continue;
        };
        let prompt: Vec<u32> = (0..40).map(|i| 100 + 37 * i).collect();
        let next: Vec<u32> = (0..40).map(|i| 200 + 11 * i).collect();
        // As they come.
        let mut cache = Cache::new();
        model.step(&cuda, &prompt, &mut cache).unwrap();
        let want: Vec<Vec<f64>> = next
            .iter()
            .map(|&t| values(&cuda, &model.step(&cuda, &[t], &mut cache).unwrap()))
            .collect();
        // Through a recording, made and made again (every few hundred
        // positions) by `recorded_step`.
        let mut cache = Cache::new();
        model.step(&cuda, &prompt, &mut cache).unwrap();
        let mut recorded = None;
        for (i, &t) in next.iter().enumerate() {
            let logits = cache
                .recorded_step(&model, &cuda, &mut recorded, t)
                .unwrap()
                .expect("the card records passes");
            assert_eq!(values(&cuda, &logits), want[i], "{name}: token {i}");
        }
        // Several sequences at once.
        let mut many = KvCache::new();
        let ids: Vec<_> = (0..3).map(|_| many.add_sequence()).collect();
        let prompts: Vec<Vec<u32>> = (0..3).map(|s| prompt[s * 5..].to_vec()).collect();
        let batch: Vec<_> = ids
            .iter()
            .zip(&prompts)
            .map(|(&id, p)| (id, &p[..]))
            .collect();
        model.step_batch(&cuda, &mut many, &batch).unwrap();
        let mut eager = KvCache::new();
        let eager_ids: Vec<_> = (0..3).map(|_| eager.add_sequence()).collect();
        let batch: Vec<_> = eager_ids
            .iter()
            .zip(&prompts)
            .map(|(&id, p)| (id, &p[..]))
            .collect();
        model.step_batch(&cuda, &mut eager, &batch).unwrap();
        let mut recorded = None;
        for (i, &t) in next.iter().take(10).enumerate() {
            let tokens = [t, t + 1, t + 2];
            let singles: Vec<[u32; 1]> = tokens.iter().map(|&t| [t]).collect();
            let entries: Vec<_> = eager_ids
                .iter()
                .zip(&singles)
                .map(|(&id, t)| (id, &t[..]))
                .collect();
            let want = values(
                &cuda,
                &model.step_batch(&cuda, &mut eager, &entries).unwrap(),
            );
            let got = match &mut recorded {
                None => {
                    let pairs: Vec<_> = ids.iter().copied().zip(tokens).collect();
                    let (step, logits) = model
                        .record_step(&cuda, &mut many, &pairs, 8)
                        .unwrap()
                        .unwrap();
                    recorded = Some(step);
                    values(&cuda, &logits)
                }
                Some(step) => {
                    let logits = model
                        .replay_step(&cuda, &mut many, step, &tokens)
                        .unwrap()
                        .unwrap();
                    values(&cuda, &logits)
                }
            };
            assert_eq!(got, want, "{name}: three sequences, token {i}");
        }
    }
}

/// The first token of a recording, computed as it comes before the pass is
/// recorded, whatever the room, against a step without a recording; then
/// the steps that follow a recording, without it.
#[test]
fn recording_changes_nothing_around_it() {
    let _alone = alone();
    let cuda = Cuda::new(0).unwrap();
    let Some(model) = model("smollm2", &cuda) else {
        return;
    };
    let prompt: Vec<u32> = (0..40).map(|i| 100 + 37 * i).collect();
    let plain = || {
        let mut cache = KvCache::new();
        let id = cache.add_sequence();
        model
            .step_batch(&cuda, &mut cache, &[(id, &prompt)])
            .unwrap();
        values(
            &cuda,
            &model
                .step_batch(&cuda, &mut cache, &[(id, &[7][..])])
                .unwrap(),
        )
    };
    let want = plain();
    for room in [0usize, 8, 100, 256] {
        let mut cache = KvCache::new();
        let id = cache.add_sequence();
        model
            .step_batch(&cuda, &mut cache, &[(id, &prompt)])
            .unwrap();
        let (step, logits) = model
            .record_step(&cuda, &mut cache, &[(id, 7)], room)
            .unwrap()
            .unwrap();
        assert_eq!(values(&cuda, &logits), want, "room {room}");
        drop(step);
        assert_eq!(plain(), want, "after a recording with room {room}");
    }
}

/// A product and an element-wise kernel, recorded, then replayed with new
/// inputs: the results of computing them as they come.
#[test]
fn a_recorded_pass_replays_with_new_inputs() {
    use kapel_core::{BF16, Tensor};
    use kapel_ops::{Matmul, Record, Run, Unary, UnaryKind, Write};
    let _alone = alone();
    let cuda = Cuda::new(0).unwrap();
    let host = |seed: usize, shape: &[usize]| {
        let values: Vec<BF16> = (0..shape.iter().product::<usize>())
            .map(|i| BF16::from_f64(((i * 7919 + seed) % 1000) as f64 / 500.0 - 1.0))
            .collect();
        Tensor::from_slice(&values, shape).unwrap()
    };
    let weights = cuda.upload(&host(1, &[256, 128])).unwrap();
    let mut x = cuda.zeros(&[4, 128], DType::BF16).unwrap();
    let pass = |x: &Tensor| {
        let y = cuda.forward(&Matmul, &[x, &weights.transpose(0, 1)?])?;
        cuda.forward(
            &Unary {
                kind: UnaryKind::Silu,
            },
            &[&y],
        )
    };
    // Once as it comes, for the tile kernel to be compiled.
    cuda.copy_from_host(&mut x, &host(2, &[4, 128])).unwrap();
    pass(&x).unwrap();
    let recording = cuda.record(&mut || pass(&x)).unwrap().unwrap();
    for seed in [3, 4] {
        let input = host(seed, &[4, 128]);
        cuda.copy_from_host(&mut x, &input).unwrap();
        recording.replay().unwrap();
        let got = values(&cuda, recording.output());
        let want = values(&cuda, &pass(&cuda.upload(&input).unwrap()).unwrap());
        assert_eq!(got, want, "seed {seed}");
    }
}
