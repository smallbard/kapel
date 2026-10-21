//! Instruction sets, detected at run time, and the number of threads.

/// A family of vector instructions the kernels can use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum IsaKind {
    /// AVX-512 (foundation, byte and word, double and quad word, vector
    /// length) with AVX2 and FMA.
    Avx512,
    /// AVX2 with FMA.
    Avx2,
    /// Nothing beyond what every target guarantees; the compiler vectorizes
    /// what it can. This is the path on ARM until NEON kernels exist.
    Scalar,
}

impl IsaKind {
    /// Whether this processor has these instructions.
    pub fn available(self) -> bool {
        match self {
            IsaKind::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            IsaKind::Avx2 => is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
            #[cfg(target_arch = "x86_64")]
            IsaKind::Avx512 => {
                IsaKind::Avx2.available()
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

/// Vector instructions that this processor is known to have.
///
/// The kernels dispatch on an `Isa` to code compiled with
/// `#[target_feature]`, which is undefined behavior on a processor without
/// those instructions. So an `Isa` can only be obtained from [`Isa::detect`]
/// or [`Isa::new`], which check the processor first, or be [`Isa::SCALAR`],
/// which every processor has: holding an `Isa` proves that its instructions
/// can run. The field is private, so safe code cannot forge one.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Isa(IsaKind);

impl Isa {
    /// No explicit vector instructions: always available.
    pub const SCALAR: Isa = Isa(IsaKind::Scalar);

    /// The best instructions of this processor.
    pub fn detect() -> Self {
        [IsaKind::Avx512, IsaKind::Avx2]
            .into_iter()
            .find_map(Isa::new)
            .unwrap_or(Isa::SCALAR)
    }

    /// The instructions `kind`, if this processor has them.
    pub fn new(kind: IsaKind) -> Option<Self> {
        kind.available().then_some(Isa(kind))
    }

    /// Which instructions these are.
    pub fn kind(self) -> IsaKind {
        self.0
    }

    /// Every instruction set this processor has, the scalar one included.
    pub fn all_available() -> impl Iterator<Item = Isa> {
        [IsaKind::Scalar, IsaKind::Avx2, IsaKind::Avx512]
            .into_iter()
            .filter_map(Isa::new)
    }
}

impl std::fmt::Debug for Isa {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// `KAPEL_THREADS` is capped at this many threads per logical processor.
const MAX_THREADS_PER_CORE: usize = 4;

/// The default number of threads: one per physical core. Simultaneous
/// multithreading shares a core's units between two threads, and brings nothing
/// to matrix products. `KAPEL_THREADS` overrides it, between 1 and four per
/// logical processor.
pub fn default_threads() -> usize {
    // Miri isolates the file system; two threads still exercise the sharing.
    if cfg!(miri) {
        return 2;
    }
    let logical = std::thread::available_parallelism().map_or(1, |n| n.get());
    if let Some(threads) = std::env::var("KAPEL_THREADS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
    {
        // More threads than this only adds scheduling overhead; a typo such
        // as `KAPEL_THREADS=80000` must not try to spawn them all.
        return threads.clamp(1, MAX_THREADS_PER_CORE * logical);
    }
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
