//! The device: context, stream, kernels, memory.

use std::any::Any;
use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::sync::{Arc, Mutex};

use cuda_core::{
    CudaContext, CudaFunction, CudaModule, CudaStream, DeviceBuffer, DriverError,
    launch_kernel_on_stream,
};
use kapel_core::{DType, Device, Error, HostStorage, Layout, Result, Storage, Tensor};
use kapel_cuda_abi::{BLOCK, MAX_RANK, View};
use kapel_ops::Transfer;

use crate::tile::Tiles;

/// The kernels, compiled to PTX by `kernels/build.sh`.
const PTX: &str = include_str!("../kernels.ptx");

/// The name of every kernel in `kernels.ptx`.
const KERNELS: &[&str] = &[
    "copy_view",
    "unary_forward",
    "unary_backward",
    "affine",
    "binary_forward",
    "binary_backward",
    "reduce_sum",
    "spread",
    "line_max",
    "place",
    "softmax_lines",
    "softmax_lines_backward",
    "norm_rows",
    "norm_rows_backward",
    "column_sum",
    "rope",
    "gather_rows",
    "scatter_rows",
];

/// Device buffers kept for reuse, by size in bytes. Freeing device memory
/// synchronizes the device; the pool avoids it at every operation. Every
/// operation runs on one stream, so a buffer given back while kernels that use
/// it are still queued can be handed out again: the next user's work is queued
/// after theirs.
type Pool = Mutex<HashMap<usize, Vec<DeviceBuffer<u8>>>>;

/// What every clone of a [`Cuda`] backend shares.
pub(crate) struct Inner {
    // Declared first, so dropped first: it borrows `stream`.
    tiles: Tiles,
    ordinal: u32,
    stream: Arc<CudaStream>,
    functions: HashMap<&'static str, CudaFunction>,
    pool: Pool,
    _module: Arc<CudaModule>,
    _context: Arc<CudaContext>,
}

/// The CUDA backend, on one device.
///
/// Operations are queued on one stream and return as soon as their kernels are
/// queued; reading a result back ([`Transfer::download`]) waits for them.
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
    /// Opens device `ordinal` and loads the kernels.
    ///
    /// # Errors
    ///
    /// If there is no such device, or its driver cannot load the kernels.
    pub fn new(ordinal: u32) -> Result<Self> {
        let op = "Cuda::new";
        let context = CudaContext::new(ordinal as usize).map_err(driver(op))?;
        let stream = context.new_stream().map_err(driver(op))?;
        let module = context.load_module_from_ptx_src(PTX).map_err(driver(op))?;
        let functions = KERNELS
            .iter()
            .map(|&name| Ok((name, module.load_function(name).map_err(driver(op))?)))
            .collect::<Result<_>>()?;
        let tiles = Tiles::new(ordinal, stream.cu_stream().cast())?;
        Ok(Self {
            inner: Arc::new(Inner {
                tiles,
                ordinal,
                stream,
                functions,
                pool: Mutex::new(HashMap::new()),
                _module: module,
                _context: context,
            }),
        })
    }

    /// The device of this backend.
    pub fn device(&self) -> Device {
        Device::Cuda(self.inner.ordinal)
    }

    /// The ordinal of the device.
    pub(crate) fn ordinal(&self) -> u32 {
        self.inner.ordinal
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

    /// Frees the buffers kept for reuse.
    pub fn clear_pool(&self) {
        self.pool().clear();
    }

    fn pool(&self) -> std::sync::MutexGuard<'_, HashMap<usize, Vec<DeviceBuffer<u8>>>> {
        self.inner.pool.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A device buffer of `bytes` bytes, from the pool if one is free. Its
    /// contents are unspecified.
    fn take(&self, bytes: usize) -> Result<DeviceBuffer<u8>> {
        if let Some(buffer) = self.pool().get_mut(&bytes).and_then(Vec::pop) {
            return Ok(buffer);
        }
        // SAFETY: the callers of `alloc` either zero the buffer or write all of
        // it before anything reads it (a kernel with one thread per element, or
        // a copy from the host).
        unsafe { DeviceBuffer::<u8>::uninitialized_async(&self.inner.stream, bytes) }
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
            inner: Arc::clone(&self.inner),
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
            inner: Arc::clone(&self.inner),
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
        let layout = Layout::contiguous(shape)?;
        let storage = self.alloc(layout.numel() * 4, zeroed)?;
        fill(storage.pointer(), layout.numel())?;
        Tensor::from_storage(Arc::new(storage), layout, DType::F32)
    }

    /// Launches kernel `name` with one thread per item of `items`.
    pub(crate) fn launch(&self, name: &'static str, items: usize, args: Args) -> Result<()> {
        if items == 0 {
            return Ok(());
        }
        let blocks = u32::try_from(items.div_ceil(BLOCK as usize))
            .map_err(|_| Error::Overflow { op: name })?;
        let function = &self.inner.functions[name];
        let mut args = args;
        let mut params = args.pointers();
        // SAFETY: `Args` holds the values in the order and with the types of the
        // kernel's parameters: each call site mirrors the kernel's signature in
        // `kernels/src/lib.rs`, a slice being a pointer then a length, and gives
        // slices that lie within their device buffers. Those buffers outlive the
        // queued kernel: they return to the pool, not to the driver, and the
        // pool only hands them to work queued after it. The 1-D grid has at
        // least one thread per item, and every kernel checks its bounds.
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
        tensor
            .storage()
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(wrong)
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
        let storage = self.storage(op, tensor)?;
        Ok(DeviceTensor {
            base: storage.pointer(),
            len: storage.len() / 4,
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

    /// A row-major copy of an `f32` tensor of this backend read through
    /// `layout` (its own, or a broadcast of it).
    pub(crate) fn materialize(
        &self,
        op: &'static str,
        tensor: &Tensor,
        layout: &Layout,
    ) -> Result<Tensor> {
        let x = self.arg_as(op, tensor, layout)?;
        self.output(layout.shape(), false, |out, n| {
            self.launch("copy_view", n, Args::new().strided(&x).slice(out, n))
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
    inner: Arc<Inner>,
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
            let bytes = buffer.len();
            self.inner
                .pool
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .entry(bytes)
                .or_default()
                .push(buffer);
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
        Device::Cuda(self.inner.ordinal)
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
}

/// One kernel parameter.
enum Arg {
    U32(u32),
    F32(f32),
    U64(u64),
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
        self.values.push(Arg::U64(pointer));
        self.values.push(Arg::U64(len as u64));
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

    /// A tensor read through a view: its whole storage, then the view.
    pub(crate) fn strided(self, x: &DeviceTensor) -> Self {
        self.slice(x.base, x.len).view(x.view)
    }

    pub(crate) fn view(mut self, v: View) -> Self {
        self.values.push(Arg::View(v));
        self
    }

    /// A pointer to each value, as `cuLaunchKernel` wants them. They stay valid
    /// while `self` lives and no value is added.
    fn pointers(&mut self) -> Vec<*mut c_void> {
        self.values
            .iter_mut()
            .map(|arg| match arg {
                Arg::U32(v) => std::ptr::from_mut(v).cast::<c_void>(),
                Arg::F32(v) => std::ptr::from_mut(v).cast::<c_void>(),
                Arg::U64(v) => std::ptr::from_mut(v).cast::<c_void>(),
                Arg::View(v) => std::ptr::from_mut(v).cast::<c_void>(),
            })
            .collect()
    }
}

/// An `f32` tensor of this backend, read through a view.
pub(crate) struct DeviceTensor {
    /// The device address of the storage's first element.
    pub(crate) base: u64,
    /// The number of `f32` elements of the storage.
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
