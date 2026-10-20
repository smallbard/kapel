//! Instruction sets, detected at run time, and the number of threads.

/// The vector instructions the kernels use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Isa {
    /// AVX-512 (foundation, byte and word, vector length) with FMA.
    Avx512,
    /// AVX2 with FMA.
    Avx2,
    /// Nothing beyond what every target guarantees; the compiler vectorizes
    /// what it can. This is the path on ARM until NEON kernels exist.
    Scalar,
}

impl Isa {
    /// The best instructions of this processor.
    pub fn detect() -> Self {
        [Isa::Avx512, Isa::Avx2]
            .into_iter()
            .find(|isa| isa.available())
            .unwrap_or(Isa::Scalar)
    }

    /// Whether this processor has these instructions.
    pub fn available(self) -> bool {
        match self {
            Isa::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => {
                Isa::Avx2.available()
                    && is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512bw")
                    && is_x86_feature_detected!("avx512dq")
                    && is_x86_feature_detected!("avx512vl")
            }
            #[cfg(not(target_arch = "x86_64"))]
            _ => false,
        }
    }
}

/// The default number of threads: one per physical core. Simultaneous
/// multithreading shares a core's units between two threads, and brings nothing
/// to matrix products. `KAPEL_THREADS` overrides it.
pub fn default_threads() -> usize {
    // Miri isolates the file system; two threads still exercise the sharing.
    if cfg!(miri) {
        return 2;
    }
    if let Some(threads) = std::env::var("KAPEL_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
    {
        return threads;
    }
    let logical = std::thread::available_parallelism().map_or(1, |n| n.get());
    physical_cores().unwrap_or(logical).clamp(1, logical)
}

/// The number of physical cores, from `/proc/cpuinfo` on Linux.
fn physical_cores() -> Option<usize> {
    let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut cores = std::collections::BTreeSet::new();
    let mut physical = None;
    for line in info.lines() {
        let mut parts = line.splitn(2, ':').map(str::trim);
        match (parts.next(), parts.next()) {
            (Some("physical id"), Some(id)) => physical = Some(id.to_string()),
            (Some("core id"), Some(id)) => {
                cores.insert((physical.clone(), id.to_string()));
            }
            _ => {}
        }
    }
    (!cores.is_empty()).then_some(cores.len())
}
