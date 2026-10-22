//! A permanent group of threads, and how work is shared between them.
//!
//! Spawning a thread costs tens of microseconds, as much as a small operation
//! of our models. So the threads are spawned once, and wait for work. A call
//! to [`Pool::run`] hands them `tasks` numbered tasks, which each thread takes
//! in turn from a shared counter, and waits until they are all done. The
//! calling thread works alongside them.
//!
//! The caller chooses how the work is split into tasks, from the shape of the
//! tensor, never from the number of threads: however the tasks end up spread
//! over the threads, each one does the same computation, and the result is
//! bit-for-bit identical.

use std::any::Any;
use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// How many rounds an idle thread busy-waits before going to sleep: about
/// 50 µs. Between two operations of a step, work comes back much sooner than
/// that, and a spinning thread resumes in a fraction of a microsecond, where a
/// sleeping one needs a system call.
const SPINS: usize = 1 << 14;

thread_local! {
    /// True inside a task: a nested call then runs in place, without
    /// redistributing the work.
    static INSIDE: Cell<bool> = const { Cell::new(false) };
}

/// A task, as the threads see it: the closure of the current call, with its
/// lifetime erased. [`Pool::run`] only returns once every thread has left the
/// task, so the closure outlives them.
#[derive(Clone, Copy)]
struct Job {
    work: *const (dyn Fn(usize) + Sync),
    tasks: usize,
}

// SAFETY: the pointer is only used during `Pool::run`, which keeps the closure
// alive (and `Sync`) until every thread is done with it.
unsafe impl Send for Job {}

/// A value alone on its cache line: the counters that the threads write do
/// not slow down the threads that read the others.
#[repr(align(64))]
struct Line<T>(T);

impl<T> std::ops::Deref for Line<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

struct Shared {
    /// The current call, on the stack of [`Pool::run`]: written before
    /// `epoch` is incremented, read by the threads once they see it, and
    /// left alone after they have all decremented `pending`.
    job: Line<AtomicPtr<Job>>,
    /// Incremented on every call: this is how a thread notices new work.
    epoch: Line<AtomicUsize>,
    /// The next task to take.
    next: Line<AtomicUsize>,
    /// The threads that have not yet finished the current call.
    pending: Line<AtomicUsize>,
    sleepers: AtomicUsize,
    wake: Condvar,
    sleep: Mutex<()>,
    stop: AtomicBool,
    /// The payload of the first task that panicked during the current call.
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

impl Shared {
    /// Takes and runs tasks until none are left.
    fn work(&self, job: Job) {
        // SAFETY: `Pool::run` waits until every thread is done.
        let work = unsafe { &*job.work };
        INSIDE.with(|inside| inside.set(true));
        loop {
            let task = self.next.fetch_add(1, Ordering::Relaxed);
            if task >= job.tasks {
                break;
            }
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| work(task))) {
                let mut first = self.panic.lock().unwrap_or_else(|e| e.into_inner());
                first.get_or_insert(payload);
            }
        }
        INSIDE.with(|inside| inside.set(false));
    }
}

/// The group of threads.
pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// One call at a time.
    calls: Mutex<()>,
}

impl Pool {
    /// A group of `threads` threads in total, the calling thread included.
    ///
    /// If the system refuses to start some of them, the group works with
    /// those it has: [`Pool::threads`] says how many. Results do not depend
    /// on it.
    pub fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared {
            job: Line(AtomicPtr::new(std::ptr::null_mut())),
            epoch: Line(AtomicUsize::new(0)),
            next: Line(AtomicUsize::new(0)),
            pending: Line(AtomicUsize::new(0)),
            sleepers: AtomicUsize::new(0),
            wake: Condvar::new(),
            sleep: Mutex::new(()),
            stop: AtomicBool::new(false),
            panic: Mutex::new(None),
        });
        let workers = (1..threads.max(1))
            .map_while(|index| {
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("kapel-{index}"))
                    .spawn(move || worker(&shared))
                    .ok()
            })
            .collect();
        Self {
            shared,
            workers,
            calls: Mutex::new(()),
        }
    }

    /// Number of threads, the calling thread included.
    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Runs `work(0)`, …, `work(tasks - 1)`, spread over the threads, and
    /// waits until they are all done.
    pub fn run(&self, tasks: usize, work: impl Fn(usize) + Sync) {
        if tasks == 0 {
            return;
        }
        if tasks == 1 || self.workers.is_empty() || INSIDE.with(Cell::get) {
            (0..tasks).for_each(work);
            return;
        }
        let _call = self.calls.lock().unwrap_or_else(|e| e.into_inner());
        let shared = &*self.shared;
        let work: &(dyn Fn(usize) + Sync) = &work;
        // SAFETY: this erases the closure's lifetime; below, we wait until no
        // thread uses it any more.
        let work: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(work) };
        let mut job = Job { work, tasks };
        // Published by the increment of `epoch` below.
        shared.job.store(&mut job, Ordering::Relaxed);
        shared.next.store(0, Ordering::Relaxed);
        shared.pending.store(self.workers.len(), Ordering::Relaxed);
        shared.epoch.fetch_add(1, Ordering::SeqCst);
        if shared.sleepers.load(Ordering::SeqCst) > 0 {
            let _guard = shared.sleep.lock().unwrap();
            shared.wake.notify_all();
        }
        shared.work(job);
        let mut spins = 0;
        while shared.pending.load(Ordering::Acquire) > 0 {
            if spins < SPINS {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
            }
        }
        let panic = shared
            .panic
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(payload) = panic {
            resume_unwind(payload);
        }
    }
}

fn worker(shared: &Shared) {
    let mut seen = 0;
    loop {
        // Wait for a new call: first busy-waiting, then asleep.
        let mut spins = 0;
        loop {
            if shared.stop.load(Ordering::Relaxed) {
                return;
            }
            if shared.epoch.load(Ordering::SeqCst) != seen {
                break;
            }
            if spins < SPINS {
                std::hint::spin_loop();
                spins += 1;
                continue;
            }
            let mut guard = shared.sleep.lock().unwrap();
            shared.sleepers.fetch_add(1, Ordering::SeqCst);
            while shared.epoch.load(Ordering::SeqCst) == seen && !shared.stop.load(Ordering::SeqCst)
            {
                guard = shared.wake.wait(guard).unwrap();
            }
            shared.sleepers.fetch_sub(1, Ordering::SeqCst);
        }
        seen = shared.epoch.load(Ordering::SeqCst);
        // SAFETY: `epoch` changed, so `run` has stored its job, which it keeps
        // on its stack until this thread decrements `pending`; this load is
        // ordered after that store by the load of `epoch`.
        let job = unsafe { *shared.job.load(Ordering::Relaxed) };
        shared.work(job);
        shared.pending.fetch_sub(1, Ordering::Release);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        {
            let _guard = self.shared.sleep.lock().unwrap();
            self.shared.wake.notify_all();
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// A pointer that can be shared between threads, so that each task writes its
/// own part of the same buffer.
#[derive(Clone, Copy)]
pub struct SharedMut<T>(*mut T);

// SAFETY: a `SharedMut` stands for a `&mut [T]` split between threads: each
// thread receives `&mut T` to its own elements, which needs `T: Send` (as
// `&mut [T]: Send` does). Access only goes through `unsafe` methods whose
// contract requires that tasks never touch overlapping parts of the buffer.
unsafe impl<T: Send> Send for SharedMut<T> {}
// SAFETY: sharing a `SharedMut` hands out the same disjoint `&mut T`, from
// several threads: `T: Send` is again what is needed, and enough.
unsafe impl<T: Send> Sync for SharedMut<T> {}

impl<T> SharedMut<T> {
    /// Shares the start of `slice`. The slice must outlive every use of the
    /// returned pointer.
    pub fn new(slice: &mut [T]) -> Self {
        Self(slice.as_mut_ptr())
    }

    /// From a pointer that is already shared.
    ///
    /// # Safety
    ///
    /// The same rules as for [`SharedMut::slice`].
    pub unsafe fn from_ptr(ptr: *mut T) -> Self {
        Self(ptr)
    }

    /// A pointer to element `offset` of the buffer.
    ///
    /// # Safety
    ///
    /// The same rules as for [`SharedMut::slice`], for everything written
    /// through it.
    pub unsafe fn ptr(&self, offset: usize) -> *mut T {
        // SAFETY: the caller guarantees that `offset` lies within the buffer.
        unsafe { self.0.add(offset) }
    }

    /// The slice `[start, start + len)` of the buffer.
    ///
    /// # Safety
    ///
    /// Two tasks must never receive overlapping slices, and the buffer must
    /// outlive the call.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice(&self, start: usize, len: usize) -> &mut [T] {
        // SAFETY: the caller guarantees that the range lies within the buffer,
        // is alive, and is not aliased by another task.
        unsafe { std::slice::from_raw_parts_mut(self.0.add(start), len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_task_runs_once() {
        let pool = Pool::new(4);
        for tasks in [0, 1, 3, 100, 1000] {
            let counts: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
            pool.run(tasks, |task| {
                counts[task].fetch_add(1, Ordering::Relaxed);
            });
            assert!(
                counts
                    .iter()
                    .all(|count| count.load(Ordering::Relaxed) == 1)
            );
        }
    }

    #[test]
    fn nested_calls_run_in_place() {
        let pool = Pool::new(3);
        let total = AtomicUsize::new(0);
        pool.run(8, |_| {
            pool.run(5, |_| {
                total.fetch_add(1, Ordering::Relaxed);
            });
        });
        assert_eq!(total.load(Ordering::Relaxed), 40);
    }

    #[test]
    fn sleeping_workers_wake_up() {
        let pool = Pool::new(4);
        std::thread::sleep(std::time::Duration::from_millis(20));
        let total = AtomicUsize::new(0);
        pool.run(64, |task| {
            total.fetch_add(task, Ordering::Relaxed);
        });
        assert_eq!(total.load(Ordering::Relaxed), 63 * 64 / 2);
    }

    #[test]
    #[should_panic(expected = "task 7 failed")]
    fn a_panicking_task_is_reported_with_its_message() {
        let pool = Pool::new(2);
        pool.run(16, |task| assert_ne!(task, 7, "task 7 failed"));
    }

    #[test]
    fn the_pool_works_after_a_panic() {
        let pool = Pool::new(3);
        let caught = catch_unwind(AssertUnwindSafe(|| {
            pool.run(16, |task| assert_ne!(task, 3));
        }));
        assert!(caught.is_err());
        let total = AtomicUsize::new(0);
        pool.run(16, |_| {
            total.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(total.load(Ordering::Relaxed), 16);
    }

    #[test]
    fn tasks_write_disjoint_parts() {
        let pool = Pool::new(4);
        let mut data = vec![0usize; 1000];
        let out = SharedMut::new(&mut data);
        pool.run(10, |task| {
            // SAFETY: each task writes its own 100 elements of `data`.
            let part = unsafe { out.slice(task * 100, 100) };
            for (i, x) in part.iter_mut().enumerate() {
                *x = task * 100 + i;
            }
        });
        assert!(data.iter().enumerate().all(|(i, &x)| i == x));
    }
}
