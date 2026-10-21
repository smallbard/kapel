//! The device: context, stream, kernels, memory.

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use cuda_core::{
    CudaContext, CudaFunction, CudaModule, CudaStream, DeviceBuffer, DriverError,
    launch_kernel_on_stream,
};
use kapel_core::{DType, Device, Error, HostStorage, Layout, Result, Storage, Tensor};
use kapel_cuda_abi::{BLOCK, MAX_RANK, View, half};
use kapel_ops::Transfer;

use crate::signatures::{self, Param};
use crate::tile::Tiles;

/// The kernels, compiled to PTX by `kernels/build.sh`.
const PTX: &str = include_str!("../kernels.ptx");

/// Locks a mutex, even if a thread panicked while holding it: every value
/// guarded here stays consistent between statements.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Buffers kept for reuse, by size in bytes. Freeing device memory
/// synchronizes the device; the pool avoids it at every operation. Every
/// operation runs on one stream, so a buffer given back while kernels that use
/// it are still queued can be handed out again: the next user's work is queued
/// after theirs.
///
/// The pool keeps every buffer given back, up to the device's memory: when an
/// allocation fails, [`take_or_alloc`] empties it and tries again, as caching
/// allocators do.
pub(crate) struct Pool<B> {
    free: HashMap<usize, Vec<B>>,
}

impl<B> Pool<B> {
    fn new() -> Self {
        Self {
            free: HashMap::new(),
        }
    }

    fn pop(&mut self, bytes: usize) -> Option<B> {
        self.free.get_mut(&bytes).and_then(Vec::pop)
    }

    fn push(&mut self, bytes: usize, buffer: B) {
        self.free.entry(bytes).or_default().push(buffer);
    }

    /// Takes every buffer out, for the caller to free outside the lock.
    fn drain(&mut self) -> Vec<B> {
        self.free.drain().flat_map(|(_, buffers)| buffers).collect()
    }
}

/// A buffer of `bytes` bytes: from the pool if one is free, otherwise from
/// `alloc`. If `alloc` fails, `release` frees what the backend keeps (the
/// pool and the cached tables) and `alloc` is tried once more, unless
/// `release` freed nothing.
fn take_or_alloc<B, E>(
    pool: &Mutex<Pool<B>>,
    bytes: usize,
    mut alloc: impl FnMut() -> std::result::Result<B, E>,
    release: impl FnOnce() -> bool,
) -> std::result::Result<B, E> {
    if let Some(buffer) = lock(pool).pop(bytes) {
        return Ok(buffer);
    }
    match alloc() {
        Ok(buffer) => Ok(buffer),
        Err(error) => {
            if release() {
                alloc()
            } else {
                Err(error)
            }
        }
    }
}

/// What the storages of a backend share with it: what identifies the backend,
/// and where their buffers go back.
pub(crate) struct Shared {
    ordinal: u32,
    pool: Mutex<Pool<DeviceBuffer<u8>>>,
}

/// Tables that depend only on shapes (the offsets of a reduction, the lines of
/// a softmax...), kept on the device so that the next operation on the same
/// shapes neither recomputes nor uploads them: an upload from pageable memory
/// waits for the stream.
struct Tables {
    map: HashMap<TableKey, Arc<Table>>,
    /// The keys, oldest first, for eviction.
    order: VecDeque<TableKey>,
    bytes: usize,
}

/// The kind of a table, and the sizes it is computed from.
type TableKey = (&'static str, Vec<usize>);

/// The most bytes of tables kept. A larger table is used once, not kept.
const TABLE_BYTES: usize = 64 << 20;

/// Values on the device, read-only once uploaded.
pub(crate) struct Table {
    pub(crate) storage: Arc<CudaStorage>,
    /// The number of values, possibly 0 (the storage then holds one).
    pub(crate) len: usize,
}

/// What every clone of a [`Cuda`] backend shares.
pub(crate) struct Inner {
    // Declared first, so dropped first: it borrows `stream`.
    tiles: Tiles,
    tables: Mutex<Tables>,
    shared: Arc<Shared>,
    stream: Arc<CudaStream>,
    functions: HashMap<&'static str, CudaFunction>,
    _module: Arc<CudaModule>,
    _context: Arc<CudaContext>,
}

/// The backends alive, one per device ordinal.
static BACKENDS: Mutex<Vec<(u32, Weak<Inner>)>> = Mutex::new(Vec::new());

/// The CUDA backend, on one device.
///
/// There is one backend per device in the process: [`Cuda::new`] on an
/// ordinal whose backend is alive returns that backend. All the work on a
/// device's tensors is thus queued on one stream, in order, and a kernel never
/// reads a buffer that another stream may still be writing.
///
/// Most operations return as soon as their kernels are queued; reading a
/// result back ([`Transfer::download`]) waits for them. Some wait for the
/// stream before returning: uploads ([`Transfer::upload`]), the first use of a
/// shape by a reduction, softmax, maximum or rotary embedding (whose tables are
/// then kept on the device), and operations that read values back to the host
/// (conversions, row gathers and scatters, the positions of rotary embeddings,
/// quantized weights, masks or normalization weights that are not `f32`).
/// Cloning the backend shares its stream and its buffers.
#[derive(Clone)]
pub struct Cuda {
    pub(crate) inner: Arc<Inner>,
}

/// Turns a driver error into a Kapel error.
pub(crate) fn driver(op: &'static str) -> impl Fn(DriverError) -> Error {
    move |e| Error::Device {
        op,
        message: e.to_string(),
    }
}

impl Cuda {
    /// The backend of device `ordinal`: the one alive in the process, or a
    /// new one, which opens the device and loads the kernels.
    ///
    /// # Errors
    ///
    /// If there is no such device, or its driver cannot load the kernels.
    pub fn new(ordinal: u32) -> Result<Self> {
        // Held while opening, so that two threads do not open two backends.
        let mut backends = lock(&BACKENDS);
        backends.retain(|(_, backend)| backend.strong_count() > 0);
        if let Some(inner) = backends
            .iter()
            .filter(|(o, _)| *o == ordinal)
            .find_map(|(_, backend)| backend.upgrade())
        {
            return Ok(Self { inner });
        }
        let cuda = Self::open(ordinal)?;
        backends.push((ordinal, Arc::downgrade(&cuda.inner)));
        Ok(cuda)
    }

    fn open(ordinal: u32) -> Result<Self> {
        let op = "Cuda::new";
        let context = CudaContext::new(ordinal as usize).map_err(driver(op))?;
        let stream = context.new_stream().map_err(driver(op))?;
        let module = context.load_module_from_ptx_src(PTX).map_err(driver(op))?;
        let functions = signatures::KERNELS
            .iter()
            .map(|&(name, _)| Ok((name, module.load_function(name).map_err(driver(op))?)))
            .collect::<Result<_>>()?;
        let tiles = Tiles::new(ordinal, stream.cu_stream().cast())?;
        Ok(Self {
            inner: Arc::new(Inner {
                tiles,
                tables: Mutex::new(Tables {
                    map: HashMap::new(),
                    order: VecDeque::new(),
                    bytes: 0,
                }),
                shared: Arc::new(Shared {
                    ordinal,
                    pool: Mutex::new(Pool::new()),
                }),
                stream,
                functions,
                _module: module,
                _context: context,
            }),
        })
    }

    /// The device of this backend.
    pub fn device(&self) -> Device {
        Device::Cuda(self.inner.shared.ordinal)
    }

    /// The ordinal of the device.
    pub(crate) fn ordinal(&self) -> u32 {
        self.inner.shared.ordinal
    }

    /// The tile runtime on this backend's stream.
    pub(crate) fn tiles(&self) -> &Tiles {
        &self.inner.tiles
    }

    /// Waits until every queued operation is done.
    ///
    /// # Errors
    ///
    /// If an operation failed on the device.
    pub fn synchronize(&self) -> Result<()> {
        self.inner
            .stream
            .synchronize()
            .map_err(driver("synchronize"))
    }

    /// Frees the buffers kept for reuse, and the tables kept on the device.
    /// Freeing waits for the device.
    pub fn clear_pool(&self) {
        self.release();
    }

    /// Frees the cached tables and the pool; whether there was anything.
    fn release(&self) -> bool {
        // Tables first: those that no operation is using go back to the pool.
        let tables = {
            let mut tables = lock(&self.inner.tables);
            tables.order.clear();
            tables.bytes = 0;
            tables.map.drain().count()
        };
        let buffers = lock(&self.inner.shared.pool).drain();
        tables > 0 || !buffers.is_empty()
    }

    /// A device buffer of `bytes` bytes, from the pool if one is free. Its
    /// contents are unspecified.
    fn take(&self, bytes: usize) -> Result<DeviceBuffer<u8>> {
        take_or_alloc(
            &self.inner.shared.pool,
            bytes,
            || {
                // SAFETY: the callers of `alloc` either zero the buffer or
                // write all of it before anything reads it (a kernel with one
                // thread per element, or a copy from the host).
                unsafe { DeviceBuffer::<u8>::uninitialized_async(&self.inner.stream, bytes) }
            },
            || self.release(),
        )
        .map_err(driver("alloc"))
    }

    /// A storage of `bytes` bytes: zeroed if `zeroed`, otherwise with
    /// unspecified contents that the caller must overwrite before reading.
    pub(crate) fn alloc(&self, bytes: usize, zeroed: bool) -> Result<CudaStorage> {
        let mut buffer = self.take(bytes)?;
        if zeroed {
            buffer
                .zero_async(&self.inner.stream)
                .map_err(driver("alloc"))?;
        }
        Ok(CudaStorage {
            buffer: Some(buffer),
            shared: Arc::clone(&self.inner.shared),
        })
    }

    /// A storage holding a copy of `bytes`.
    pub(crate) fn upload_bytes(&self, bytes: &[u8]) -> Result<CudaStorage> {
        let mut buffer = self.take(bytes.len())?;
        if !bytes.is_empty() {
            buffer
                .copy_from_host(&self.inner.stream, bytes)
                .map_err(driver("upload"))?;
        }
        Ok(CudaStorage {
            buffer: Some(buffer),
            shared: Arc::clone(&self.inner.shared),
        })
    }

    /// Integers on the device, for index and offset arguments. An empty list
    /// becomes one zero, so that the pointer is valid; pass the true length to
    /// the kernel.
    pub(crate) fn upload_u32(&self, values: &[u32]) -> Result<CudaStorage> {
        let values = if values.is_empty() { &[0][..] } else { values };
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.upload_bytes(&bytes)
    }

    /// Table `kind` for `key`: the one kept on the device, or `make`'s values,
    /// uploaded (which waits for the stream) and kept.
    pub(crate) fn table(
        &self,
        kind: &'static str,
        key: &[usize],
        make: impl FnOnce() -> Result<Vec<u32>>,
    ) -> Result<Arc<Table>> {
        let key = (kind, key.to_vec());
        if let Some(table) = lock(&self.inner.tables).map.get(&key) {
            return Ok(Arc::clone(table));
        }
        let values = make()?;
        let table = Arc::new(Table {
            storage: Arc::new(self.upload_u32(&values)?),
            len: values.len(),
        });
        let bytes = table.storage.len();
        if bytes <= TABLE_BYTES {
            let mut tables = lock(&self.inner.tables);
            while tables.bytes + bytes > TABLE_BYTES {
                let Some(old) = tables.order.pop_front() else {
                    break;
                };
                // A table still in use goes back to the pool when its last
                // user, whose kernels are queued, drops it.
                if let Some(old) = tables.map.remove(&old) {
                    tables.bytes -= old.storage.len();
                }
            }
            if tables.map.insert(key.clone(), Arc::clone(&table)).is_none() {
                tables.order.push_back(key);
                tables.bytes += bytes;
            }
        }
        Ok(table)
    }

    /// A contiguous `f32` tensor of `shape` whose elements are all `value`,
    /// kept on the device like a table.
    pub(crate) fn filled(&self, op: &'static str, value: f32, shape: &[usize]) -> Result<Tensor> {
        let layout = Layout::contiguous(shape)?;
        let n = layout.numel();
        let table = self.table("filled", &[n, value.to_bits() as usize], || {
            if n > u32::MAX as usize {
                return Err(Error::Overflow { op });
            }
            Ok(vec![value.to_bits(); n])
        })?;
        Tensor::from_storage(
            Arc::clone(&table.storage) as Arc<dyn Storage>,
            layout,
            DType::F32,
        )
    }

    /// Floats on the device, as a contiguous tensor of `shape`.
    pub(crate) fn upload_f32(&self, values: &[f32], shape: &[usize]) -> Result<Tensor> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let storage = self.upload_bytes(&bytes)?;
        Tensor::from_storage(Arc::new(storage), Layout::contiguous(shape)?, DType::F32)
    }

    /// An `f32` tensor of `shape`, zeroed if `zeroed`, filled by `fill`, which
    /// receives the device pointer and the number of elements.
    pub(crate) fn output(
        &self,
        shape: &[usize],
        zeroed: bool,
        fill: impl FnOnce(u64, usize) -> Result<()>,
    ) -> Result<Tensor> {
        self.output_as(shape, DType::F32, zeroed, fill)
    }

    /// A tensor of `shape` and `dtype` (with elements of their own), as
    /// [`Cuda::output`].
    pub(crate) fn output_as(
        &self,
        shape: &[usize],
        dtype: DType,
        zeroed: bool,
        fill: impl FnOnce(u64, usize) -> Result<()>,
    ) -> Result<Tensor> {
        let layout = Layout::contiguous(shape)?;
        let bytes = dtype.storage_bytes(layout.numel())?;
        let storage = self.alloc(bytes, zeroed)?;
        fill(storage.pointer(), layout.numel())?;
        Tensor::from_storage(Arc::new(storage), layout, dtype)
    }

    /// Launches kernel `name` with one thread per item of `items`.
    ///
    /// # Errors
    ///
    /// [`Error::Overflow`] if there are more than `u32::MAX` items, or a slice
    /// longer than `u32::MAX` elements: the kernels compute positions in `u32`.
    pub(crate) fn launch(&self, name: &'static str, items: usize, args: Args) -> Result<()> {
        let mismatch = || Error::Device {
            op: name,
            message: "internal error: the arguments do not match the kernel's parameters".into(),
        };
        let expected = signatures::of(name).ok_or_else(mismatch)?;
        if !args.kinds().eq(expected.iter().copied()) {
            return Err(mismatch());
        }
        if items > u32::MAX as usize || args.too_long {
            return Err(Error::Overflow { op: name });
        }
        if items == 0 {
            return Ok(());
        }
        let blocks = u32::try_from(items.div_ceil(BLOCK as usize))
            .map_err(|_| Error::Overflow { op: name })?;
        let function = self.inner.functions.get(name).ok_or_else(mismatch)?;
        let mut args = args;
        let mut params = args.pointers();
        // SAFETY: `Args` holds the values in the order and with the types of the
        // kernel's parameters: checked above against `signatures::KERNELS`,
        // which a test checks against the PTX's parameter lists. Each slice
        // lies within its device buffer (`Args::slice` is given a buffer's
        // pointer and at most its length), and no slice nor the grid exceeds
        // `u32::MAX` elements (checked above), so the kernels' `u32` positions
        // do not wrap. The buffers outlive the queued kernel: they return to
        // the pool, not to the driver, and the pool only hands them to work
        // queued after it. The 1-D grid has at least one thread per item.
        // Every kernel reads through bounds-checked indexing, and writes either
        // through `DisjointSlice::get_mut` at its own index or, for a line or
        // row of its own, through a raw pointer after checking the line's
        // bounds: no write leaves its buffer, and no two threads write the same
        // element (see `kernels/src/lib.rs`).
        unsafe {
            launch_kernel_on_stream(
                function,
                (blocks, 1, 1),
                (BLOCK, 1, 1),
                0,
                &self.inner.stream,
                &mut params,
            )
            .map_err(driver(name))
        }
    }

    /// The storage of a tensor of this backend.
    ///
    /// # Errors
    ///
    /// [`Error::WrongDevice`] if the tensor is on another device, or on this
    /// one but not a storage of this backend (only possible if another
    /// library opened the device): its work could run on another stream, in
    /// any order with this one's.
    pub(crate) fn storage<'t>(
        &self,
        op: &'static str,
        tensor: &'t Tensor,
    ) -> Result<&'t CudaStorage> {
        let wrong = || Error::WrongDevice {
            op,
            expected: self.device(),
            got: tensor.device(),
        };
        if tensor.device() != self.device() {
            return Err(wrong());
        }
        let storage = tensor
            .storage()
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(wrong)?;
        if !Arc::ptr_eq(&storage.shared, &self.inner.shared) {
            return Err(Error::Device {
                op,
                message: "the tensor belongs to another CUDA backend of the same device".into(),
            });
        }
        Ok(storage)
    }

    /// An `f32` tensor of this backend as a kernel argument, read through
    /// `layout` (its own, or a broadcast of it).
    pub(crate) fn arg_as(
        &self,
        op: &'static str,
        tensor: &Tensor,
        layout: &Layout,
    ) -> Result<DeviceTensor> {
        if tensor.dtype() != DType::F32 {
            return Err(Error::UnsupportedDType {
                op,
                dtype: tensor.dtype(),
            });
        }
        self.any_arg(op, tensor, layout)
    }

    /// A tensor of this backend, of any type with elements of their own (not
    /// quantized blocks), as a kernel argument read through `layout`. The
    /// storage's length counts its elements.
    fn any_arg(&self, op: &'static str, tensor: &Tensor, layout: &Layout) -> Result<DeviceTensor> {
        let dtype = tensor.dtype();
        let size = dtype
            .element_size()
            .ok_or(Error::UnsupportedDType { op, dtype })?;
        let storage = self.storage(op, tensor)?;
        Ok(DeviceTensor {
            base: storage.pointer(),
            len: storage.len() / size,
            view: view_of(op, layout)?,
        })
    }

    /// An `f32` tensor of this backend as a kernel argument.
    pub(crate) fn arg(&self, op: &'static str, tensor: &Tensor) -> Result<DeviceTensor> {
        self.arg_as(op, tensor, tensor.layout())
    }

    /// An `f32` tensor of this backend with its elements in row-major order:
    /// itself if it already is, otherwise a copy made on the device.
    pub(crate) fn contiguous(&self, op: &'static str, tensor: &Tensor) -> Result<Tensor> {
        if tensor.is_contiguous() {
            self.arg(op, tensor)?;
            return Ok(tensor.clone());
        }
        self.materialize(op, tensor, tensor.layout())
    }

    /// A row-major copy of a tensor of this backend read through `layout` (its
    /// own, or a broadcast of it), bit for bit, in its own type.
    pub(crate) fn materialize(
        &self,
        op: &'static str,
        tensor: &Tensor,
        layout: &Layout,
    ) -> Result<Tensor> {
        let dtype = tensor.dtype();
        let x = self.any_arg(op, tensor, layout)?;
        let kernel = match dtype.element_size() {
            Some(1) => "copy_view_8",
            Some(2) => "copy_view_16",
            Some(4) => "copy_view_32",
            _ => "copy_view_64",
        };
        self.output_as(layout.shape(), dtype, false, |out, n| {
            self.launch(kernel, n, Args::new().strided(&x).slice(out, n))
        })
    }

    /// A float tensor of this backend in `f32`, read through its layout: an
    /// `f32` tensor as it is, a `bf16` or `f16` one widened (exactly) into a
    /// row-major copy.
    pub(crate) fn widened(&self, op: &'static str, tensor: &Tensor) -> Result<Tensor> {
        let kind = match tensor.dtype() {
            DType::BF16 => half::BF16,
            DType::F16 => half::F16,
            _ => {
                self.storage(op, tensor)?;
                return Ok(tensor.clone());
            }
        };
        let x = self.any_arg(op, tensor, tensor.layout())?;
        self.output(tensor.shape(), false, |out, n| {
            self.launch(
                "widen_16",
                n,
                Args::new().u32(kind).strided(&x).slice(out, n),
            )
        })
    }

    /// An `f32` tensor of this backend rounded to `dtype` (ties to even, as
    /// [`DType::encode`] does): itself for `f32`, a row-major copy for `bf16`
    /// and `f16`.
    pub(crate) fn narrowed(
        &self,
        op: &'static str,
        tensor: &Tensor,
        dtype: DType,
    ) -> Result<Tensor> {
        let kind = match dtype {
            DType::F32 => {
                self.storage(op, tensor)?;
                return Ok(tensor.clone());
            }
            DType::BF16 => half::BF16,
            DType::F16 => half::F16,
            dtype => return Err(Error::UnsupportedDType { op, dtype }),
        };
        let x = self.flat(op, tensor)?;
        self.output_as(tensor.shape(), dtype, false, |out, n| {
            self.launch("narrow_16", n, Args::new().u32(kind).flat(&x).slice(out, n))
        })
    }

    /// An `f32` tensor of this backend as a row-major slice argument.
    pub(crate) fn flat(&self, op: &'static str, tensor: &Tensor) -> Result<Flat> {
        let tensor = self.contiguous(op, tensor)?;
        let storage = self.storage(op, &tensor)?;
        let ptr = storage.pointer() + tensor.layout().offset() as u64 * 4;
        Ok(Flat {
            ptr,
            len: tensor.numel(),
            _tensor: tensor,
        })
    }
}

impl fmt::Debug for Cuda {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cuda")
            .field("device", &self.device())
            .finish()
    }
}

impl Transfer for Cuda {
    fn upload(&self, tensor: &Tensor) -> Result<Tensor> {
        match tensor.device() {
            Device::Cpu => {}
            device if device == self.device() => return Ok(tensor.clone()),
            device => {
                return Err(Error::WrongDevice {
                    op: "upload",
                    expected: Device::Cpu,
                    got: device,
                });
            }
        }
        let tensor = tensor.contiguous()?;
        let bytes = tensor.storage().host_bytes().ok_or(Error::WrongDevice {
            op: "upload",
            expected: Device::Cpu,
            got: tensor.device(),
        })?;
        let dtype = tensor.dtype();
        let start = dtype.storage_bytes(tensor.layout().offset())?;
        let len = dtype.storage_bytes(tensor.numel())?;
        let storage = self.upload_bytes(&bytes[start..start + len])?;
        Tensor::from_storage(
            Arc::new(storage),
            Layout::contiguous(tensor.shape())?,
            dtype,
        )
    }

    fn download(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.device() == Device::Cpu {
            return Ok(tensor.clone());
        }
        let storage = self.storage("download", tensor)?;
        let bytes = match &storage.buffer {
            Some(buffer) if !buffer.is_empty() => buffer
                .to_host_vec(&self.inner.stream)
                .map_err(driver("download"))?,
            _ => Vec::new(),
        };
        Tensor::from_storage(
            Arc::new(HostStorage::from_bytes(&bytes)),
            tensor.layout().clone(),
            tensor.dtype(),
        )
    }
}

/// The bytes of a tensor on the device, given back to the pool when dropped.
pub(crate) struct CudaStorage {
    buffer: Option<DeviceBuffer<u8>>,
    shared: Arc<Shared>,
}

impl CudaStorage {
    /// The device address of the first byte.
    pub(crate) fn pointer(&self) -> u64 {
        self.buffer.as_ref().map_or(0, DeviceBuffer::cu_deviceptr)
    }
}

impl Drop for CudaStorage {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            lock(&self.shared.pool).push(buffer.len(), buffer);
        }
    }
}

impl fmt::Debug for CudaStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaStorage")
            .field("bytes", &self.len())
            .finish()
    }
}

impl Storage for CudaStorage {
    fn device(&self) -> Device {
        Device::Cuda(self.shared.ordinal)
    }

    fn len(&self) -> usize {
        self.buffer.as_ref().map_or(0, DeviceBuffer::len)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The arguments of a kernel launch, in the order of its parameters.
#[derive(Default)]
pub(crate) struct Args {
    values: Vec<Arg>,
    /// Whether a slice is longer than `u32::MAX` elements.
    too_long: bool,
}

/// One kernel argument.
enum Arg {
    U32(u32),
    F32(f32),
    /// A pointer and a length: two kernel parameters.
    Slice([u64; 2]),
    View(View),
}

impl Args {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn u32(mut self, v: u32) -> Self {
        self.values.push(Arg::U32(v));
        self
    }

    pub(crate) fn f32(mut self, v: f32) -> Self {
        self.values.push(Arg::F32(v));
        self
    }

    /// A slice: its device pointer, then its length in elements.
    pub(crate) fn slice(mut self, pointer: u64, len: usize) -> Self {
        self.too_long |= len > u32::MAX as usize;
        self.values.push(Arg::Slice([pointer, len as u64]));
        self
    }

    /// A row-major tensor as a slice.
    pub(crate) fn flat(self, x: &Flat) -> Self {
        self.slice(x.ptr, x.len)
    }

    /// Values on the device as a slice of `len` elements.
    pub(crate) fn list(self, x: &CudaStorage, len: usize) -> Self {
        self.slice(x.pointer(), len)
    }

    /// A table as a slice.
    pub(crate) fn table(self, x: &Table) -> Self {
        self.slice(x.storage.pointer(), x.len)
    }

    /// A tensor read through a view: its whole storage, then the view.
    pub(crate) fn strided(self, x: &DeviceTensor) -> Self {
        self.slice(x.base, x.len).view(x.view)
    }

    pub(crate) fn view(mut self, v: View) -> Self {
        self.values.push(Arg::View(v));
        self
    }

    /// The kind of each argument, to check against the kernel's.
    fn kinds(&self) -> impl Iterator<Item = Param> + '_ {
        self.values.iter().map(|arg| match arg {
            Arg::U32(_) => Param::U32,
            Arg::F32(_) => Param::F32,
            Arg::Slice(_) => Param::Slice,
            Arg::View(_) => Param::View,
        })
    }

    /// A pointer to each kernel parameter, as `cuLaunchKernel` wants them.
    /// They stay valid while `self` lives and no value is added.
    fn pointers(&mut self) -> Vec<*mut c_void> {
        let mut pointers = Vec::with_capacity(self.values.len() + 4);
        for arg in &mut self.values {
            match arg {
                Arg::U32(v) => pointers.push(std::ptr::from_mut(v).cast::<c_void>()),
                Arg::F32(v) => pointers.push(std::ptr::from_mut(v).cast::<c_void>()),
                Arg::Slice([pointer, len]) => {
                    pointers.push(std::ptr::from_mut(pointer).cast::<c_void>());
                    pointers.push(std::ptr::from_mut(len).cast::<c_void>());
                }
                Arg::View(v) => pointers.push(std::ptr::from_mut(v).cast::<c_void>()),
            }
        }
        pointers
    }
}

/// A tensor of this backend, read through a view.
pub(crate) struct DeviceTensor {
    /// The device address of the storage's first element.
    pub(crate) base: u64,
    /// The number of elements of the storage.
    pub(crate) len: usize,
    /// How the tensor reads its storage.
    pub(crate) view: View,
}

/// An `f32` tensor of this backend in row-major order, as a slice.
pub(crate) struct Flat {
    /// The device address of the first element.
    pub(crate) ptr: u64,
    /// The number of elements.
    pub(crate) len: usize,
    /// Keeps the storage alive.
    _tensor: Tensor,
}

/// The view of a layout, padded to [`MAX_RANK`] dimensions with leading 1s.
pub(crate) fn view_of(op: &'static str, layout: &Layout) -> Result<View> {
    if layout.rank() > MAX_RANK {
        return Err(Error::InvalidArgument {
            op,
            message: format!(
                "at most {MAX_RANK} dimensions on the device, got {}",
                layout.rank()
            ),
        });
    }
    let mut view = View {
        shape: [1; MAX_RANK],
        strides: [0; MAX_RANK],
        offset: small(op, layout.offset())?,
    };
    let lead = MAX_RANK - layout.rank();
    for (i, (&n, &s)) in layout.shape().iter().zip(layout.strides()).enumerate() {
        view.shape[lead + i] = small(op, n)?;
        view.strides[lead + i] = small(op, s)?;
    }
    Ok(view)
}

/// A size or an offset as a kernel integer.
pub(crate) fn small(op: &'static str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Overflow { op })
}

/// Sizes or offsets as kernel integers.
pub(crate) fn smalls(op: &'static str, values: &[usize]) -> Result<Vec<u32>> {
    values.iter().map(|&v| small(op, v)).collect()
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn a_failed_allocation_empties_the_pool_and_retries_once() {
        let pool = Mutex::new(Pool::new());
        lock(&pool).push(8, vec![0u8; 8]);
        let calls = Cell::new(0);
        // The first attempt fails, as when the device is full; the second,
        // after the pool is emptied, succeeds.
        let got = take_or_alloc(
            &pool,
            16,
            || {
                calls.set(calls.get() + 1);
                if calls.get() == 1 {
                    Err("full")
                } else {
                    Ok(vec![0u8; 16])
                }
            },
            || !lock(&pool).drain().is_empty(),
        );
        assert_eq!(got.map(|b| b.len()), Ok(16));
        assert_eq!(calls.get(), 2);
        assert!(lock(&pool).pop(8).is_none());
    }

    #[test]
    fn a_failed_allocation_with_nothing_to_free_fails() {
        let pool: Mutex<Pool<Vec<u8>>> = Mutex::new(Pool::new());
        let calls = Cell::new(0);
        let got = take_or_alloc(
            &pool,
            16,
            || {
                calls.set(calls.get() + 1);
                Err::<Vec<u8>, _>("full")
            },
            || !lock(&pool).drain().is_empty(),
        );
        assert_eq!(got, Err("full"));
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn a_free_buffer_of_the_size_is_reused() {
        let pool = Mutex::new(Pool::new());
        lock(&pool).push(16, vec![1u8; 16]);
        let got = take_or_alloc(&pool, 16, || Err("unused"), || false);
        assert_eq!(got, Ok(vec![1u8; 16]));
    }
}
