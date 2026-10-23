//! Recorded passes: CUDA graphs.
//!
//! While a pass is recorded, the backend's stream captures the kernels it
//! queues instead of running them, and its buffers come from an arena of the
//! recording (see `device::Arena`): they stay its own while it lives, since
//! every replay reads and writes them again. The graph is then instantiated
//! once, and each replay launches all its kernels in one call of the driver.

use std::sync::Arc;

use cuda_bindings::{CUgraph, CUgraphExec};
use kapel_core::{Error, Result, Tensor};
use kapel_ops::{Record, Recording};

use crate::device::{Arena, Cuda};

/// The name of recordings in errors.
const RECORD: &str = "record";

/// An error for a driver call that returned `status`.
fn failed(what: &str, status: cuda_bindings::CUresult) -> Error {
    Error::Device {
        op: RECORD,
        message: format!("{what}: error {status}"),
    }
}

/// An instantiated graph, with what it reads and writes.
struct Graph {
    exec: CUgraphExec,
    graph: CUgraph,
    cuda: Cuda,
    _arena: Arc<Arena>,
}

// SAFETY: the driver's handles may be used from any thread of the process;
// launches go to the backend's stream, whose work is ordered whatever the
// thread that queues it.
unsafe impl Send for Graph {}
// SAFETY: as above; launching takes the handles by value, and nothing else
// is done with them until the graph is dropped.
unsafe impl Sync for Graph {}

impl Graph {
    fn launch(&self) -> Result<()> {
        // SAFETY: `exec` is a live instantiation of a graph whose buffers
        // `_arena` keeps; the launch is queued on the backend's stream.
        let status = unsafe { cuda_bindings::cuGraphLaunch(self.exec, self.cuda.raw_stream()) };
        if status != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            return Err(failed("launch", status));
        }
        Ok(())
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: the handles are live and dropped once; a replay still
        // queued keeps running, the driver frees the graph after it.
        unsafe {
            cuda_bindings::cuGraphExecDestroy(self.exec);
            cuda_bindings::cuGraphDestroy(self.graph);
        }
    }
}

impl Record for Cuda {
    fn records(&self) -> bool {
        true
    }

    /// Captures the backend's stream while `pass` queues its work, in relaxed
    /// mode: the pass may allocate (outside the stream) and look up the
    /// backend's tables, but it must not wait for the device nor copy from
    /// the host. The first pass of a given shape should run once before it
    /// is recorded: its tile kernels are then compiled, and its tables made.
    ///
    /// The stream is the device's only one: while a pass records, no other
    /// thread may use the backend, whose work would be recorded too.
    fn record(&self, pass: &mut dyn FnMut() -> Result<Tensor>) -> Result<Option<Recording>> {
        let arena = Arc::new(Arena::new(self));
        let stream = self.raw_stream();
        // SAFETY: the stream is the backend's, not capturing (the lock on
        // `capture` below is taken by one recording at a time).
        let status = unsafe {
            cuda_bindings::cuStreamBeginCapture_v2(
                stream,
                cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED,
            )
        };
        if status != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            return Err(failed("begin capture", status));
        }
        *self.inner.capture.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::clone(&arena));
        let output = pass();
        *self.inner.capture.lock().unwrap_or_else(|e| e.into_inner()) = None;
        // The tables it kept hold the arena: let them go.
        arena
            .angles
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        let mut graph: CUgraph = std::ptr::null_mut();
        // SAFETY: the stream is capturing, begun above; ending it is
        // required even when the pass failed.
        let status = unsafe { cuda_bindings::cuStreamEndCapture(stream, &mut graph) };
        let output = match output {
            Ok(output) if status == cuda_bindings::cudaError_enum_CUDA_SUCCESS => output,
            other => {
                if !graph.is_null() {
                    // SAFETY: the graph was captured above and is not used.
                    unsafe { cuda_bindings::cuGraphDestroy(graph) };
                }
                // A recording that failed for lack of memory may succeed
                // once the buffers kept for reuse are freed.
                drop(arena);
                self.clear_pool();
                return Err(match other {
                    Err(e) => e,
                    Ok(_) => failed("end capture", status),
                });
            }
        };
        let mut exec: CUgraphExec = std::ptr::null_mut();
        // SAFETY: `graph` is the graph just captured.
        let status = unsafe { cuda_bindings::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        if status != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            // SAFETY: the graph was made above and is not used again.
            unsafe { cuda_bindings::cuGraphDestroy(graph) };
            return Err(failed("instantiate", status));
        }
        let graph = Arc::new(Graph {
            exec,
            graph,
            cuda: self.clone(),
            _arena: arena,
        });
        Ok(Some(Recording::new(
            output,
            Box::new(move || graph.launch()),
        )))
    }
}
