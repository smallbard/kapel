//! Recorded passes on the card (CUDA graphs): a model's next tokens computed
//! through replays of a recorded step give the logits of computing each step
//! as it comes, bit for bit, for one sequence and for several. Uses the small
//! models of `data/parity`; without them, the test is skipped.

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use kapel_core::DType;
use kapel_cuda::Cuda;
use kapel_hub::Checkpoint;
use kapel_models::{Cache, KvCache, Model, PAGE, Recorded};
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
            let pairs: Vec<_> = ids.iter().copied().zip(tokens).collect();
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
                    // Four rows, the last one padding.
                    let pages = (many.len(ids[0]) + 9).div_ceil(PAGE);
                    let Recorded::Made(step, logits) = model
                        .record_step(&cuda, &mut many, &pairs, 4, pages)
                        .unwrap()
                    else {
                        panic!("the card records passes")
                    };
                    recorded = Some(step);
                    values(&cuda, &logits)
                }
                Some(step) => {
                    let logits = model
                        .replay_step(&cuda, &mut many, step, &pairs)
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
        let Recorded::Made(step, logits) = model
            .record_step(&cuda, &mut cache, &[(id, 7)], 1, (41 + room).div_ceil(PAGE))
            .unwrap()
        else {
            panic!("the card records passes")
        };
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

/// A recording of eight rows serves three sequences, then two of them, in a
/// cache of fixed size: the padding rows (five, then six) write lines of
/// their own, and the logits are those of steps as they come.
#[test]
fn a_recording_serves_smaller_batches() {
    let _alone = alone();
    let cuda = Cuda::new(0).unwrap();
    let Some(model) = model("smollm2", &cuda) else {
        return;
    };
    let prompts: Vec<Vec<u32>> = (0..3)
        .map(|s| (0..20 + 7 * s).map(|i| 100 + 37 * i + s).collect())
        .collect();
    let start = |cache: &mut KvCache| {
        let ids: Vec<_> = prompts.iter().map(|_| cache.add_sequence()).collect();
        let batch: Vec<_> = ids
            .iter()
            .zip(&prompts)
            .map(|(&id, p)| (id, &p[..]))
            .collect();
        model.step_batch(&cuda, cache, &batch).unwrap();
        ids
    };
    let (mut eager, mut many) = (KvCache::with_pages(64), KvCache::with_pages(64));
    let eager_ids = start(&mut eager);
    let ids = start(&mut many);
    let mut recorded = None;
    for i in 0..6u32 {
        // Three sequences, then the first two only.
        let n = if i < 3 { 3 } else { 2 };
        let tokens: Vec<u32> = (0..n).map(|s| 300 + 13 * i + s).collect();
        let singles: Vec<[u32; 1]> = tokens.iter().map(|&t| [t]).collect();
        let entries: Vec<_> = eager_ids[..n as usize]
            .iter()
            .zip(&singles)
            .map(|(&id, t)| (id, &t[..]))
            .collect();
        let want = values(
            &cuda,
            &model.step_batch(&cuda, &mut eager, &entries).unwrap(),
        );
        let pairs: Vec<_> = ids[..n as usize].iter().copied().zip(tokens).collect();
        let got = match &mut recorded {
            None => {
                let Recorded::Made(step, logits) =
                    model.record_step(&cuda, &mut many, &pairs, 8, 8).unwrap()
                else {
                    panic!("the card records passes")
                };
                recorded = Some(step);
                values(&cuda, &logits)
            }
            Some(step) => values(
                &cuda,
                &model
                    .replay_step(&cuda, &mut many, step, &pairs)
                    .unwrap()
                    .expect("the recording fits"),
            ),
        };
        assert_eq!(got, want, "step {i}, {n} sequences");
    }
}

/// The tables a recording reads stay its own: once the backend lets go of
/// its tables and buffers (as when memory runs short), and the memory is
/// written over, replays still give the logits of steps as they come.
#[test]
fn a_recording_keeps_the_tables_it_reads() {
    use kapel_core::Tensor;
    let _alone = alone();
    let cuda = Cuda::new(0).unwrap();
    let Some(model) = model("smollm2", &cuda) else {
        return;
    };
    let prompt: Vec<u32> = (0..30).map(|i| 100 + 37 * i).collect();
    let start = |cache: &mut KvCache| {
        let id = cache.add_sequence();
        model.step_batch(&cuda, cache, &[(id, &prompt)]).unwrap();
        id
    };
    let (mut eager, mut recorded_cache) = (KvCache::with_pages(16), KvCache::with_pages(16));
    let (e, r) = (start(&mut eager), start(&mut recorded_cache));
    let Recorded::Made(mut step, _) = model
        .record_step(&cuda, &mut recorded_cache, &[(r, 5)], 1, 8)
        .unwrap()
    else {
        panic!("the card records passes")
    };
    model
        .step_batch(&cuda, &mut eager, &[(e, &[5][..])])
        .unwrap();
    cuda.clear_pool();
    // Memory taken again, and written over.
    let junk: Vec<Tensor> = (8..22)
        .flat_map(|bits| {
            let cuda = &cuda;
            (0..4).map(move |_| {
                let values = vec![u32::MAX; 1 << bits];
                cuda.upload(&Tensor::from_slice(&values, &[values.len()]).unwrap())
                    .unwrap()
            })
        })
        .collect();
    for i in 0..4u32 {
        let token = 40 + i;
        let want = values(
            &cuda,
            &model
                .step_batch(&cuda, &mut eager, &[(e, &[token][..])])
                .unwrap(),
        );
        let got = model
            .replay_step(&cuda, &mut recorded_cache, &mut step, &[(r, token)])
            .unwrap()
            .expect("the recording fits");
        assert_eq!(values(&cuda, &got), want, "token {i}");
    }
    drop(junk);
}
