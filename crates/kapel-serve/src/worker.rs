//! The thread that runs the engine, and how requests reach it.
//!
//! The handlers prepare their requests (prompt, constraint) on their own
//! threads, then send them to the engine's thread. It admits them between two
//! passes, and sends each token to the request it belongs to, through a
//! channel that the handler reads. A handler that goes away (its client
//! closed the connection) closes its channel: the engine notices before its
//! next pass, and cancels the request.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;

use kapel_infer::{
    Engine, EngineOptions, Finish, InferError, Piece, Pipeline, Prepared, RequestId,
};
use kapel_models::ModelError;
use kapel_ops::{Backend, Transfer};
use tokio::sync::mpsc::UnboundedSender;

use crate::log;
use crate::metrics::Metrics;

/// What the engine sends back for one answer of a request.
#[derive(Debug)]
pub(crate) enum Output {
    /// A token, and the text it makes final.
    Piece(Piece),
    /// The answer ended; the prompt positions taken from shared pages.
    Done { finish: Finish, cached: usize },
    /// The pass failed.
    Failed(String),
}

/// One answer to compute.
pub(crate) struct Submission {
    pub(crate) prepared: Prepared,
    /// Which answer of its request it is (`n` answers each have one).
    pub(crate) index: usize,
    pub(crate) out: UnboundedSender<(usize, Output)>,
}

/// The handlers' side of the engine's thread.
#[derive(Clone)]
pub(crate) struct Worker {
    submit: mpsc::Sender<Submission>,
    /// Answers sent and not yet ended.
    in_flight: Arc<AtomicUsize>,
    alive: Arc<AtomicBool>,
}

impl Worker {
    /// Starts the engine's thread on `pipeline`.
    pub(crate) fn start<B>(
        pipeline: Arc<Pipeline<B>>,
        options: EngineOptions,
        metrics: Arc<Metrics>,
    ) -> Self
    where
        B: Backend + Transfer + Send + Sync + 'static,
    {
        let (submit, received) = mpsc::channel();
        let in_flight = Arc::new(AtomicUsize::new(0));
        let alive = Arc::new(AtomicBool::new(true));
        let worker = Self {
            submit,
            in_flight: Arc::clone(&in_flight),
            alive: Arc::clone(&alive),
        };
        std::thread::Builder::new()
            .name("kapel-engine".into())
            .spawn(move || {
                let mut engine = pipeline.engine(options);
                run(
                    &mut engine,
                    &pipeline,
                    &received,
                    &in_flight,
                    &metrics,
                    &alive,
                );
                alive.store(false, Ordering::Relaxed);
                log::event("engine_stopped", serde_json::Map::new());
            })
            .expect("a thread for the engine");
        worker
    }

    /// Sends answers to the engine; `false` if its thread has stopped.
    pub(crate) fn submit(&self, submissions: Vec<Submission>) -> bool {
        let count = submissions.len();
        self.in_flight.fetch_add(count, Ordering::Relaxed);
        for (i, submission) in submissions.into_iter().enumerate() {
            if self.submit.send(submission).is_err() {
                self.in_flight.fetch_sub(count - i, Ordering::Relaxed);
                return false;
            }
        }
        true
    }

    /// Answers sent and not yet ended.
    pub(crate) fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }

    /// Whether the engine's thread runs, on a device that has not failed.
    pub(crate) fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }
}

/// An answer in the engine, and where its tokens go.
struct Client {
    index: usize,
    out: UnboundedSender<(usize, Output)>,
}

/// Whether `error` came from the device itself (not from the cache, nor a
/// request).
fn is_device_error(error: &InferError) -> bool {
    use kapel_core::Error;
    matches!(
        error,
        InferError::Tensor(Error::Device { .. })
            | InferError::Model(ModelError::Tensor(Error::Device { .. }))
    )
}

/// Whether the device still computes: a value sent and read back.
fn device_answers<B: Backend + Transfer>(pipeline: &Pipeline<B>) -> bool {
    let backend = pipeline.backend();
    kapel_core::Tensor::from_slice(&[7u32], &[1])
        .and_then(|t| backend.upload(&t))
        .and_then(|t| backend.download(&t))
        .and_then(|t| t.to_vec::<u32>())
        .is_ok_and(|v| v == [7])
}

fn run<B: Backend + Transfer>(
    engine: &mut Engine<'_, B>,
    pipeline: &Pipeline<B>,
    received: &mpsc::Receiver<Submission>,
    in_flight: &AtomicUsize,
    metrics: &Metrics,
    healthy: &AtomicBool,
) {
    let mut clients: HashMap<RequestId, Client> = HashMap::new();
    let admit =
        |engine: &mut Engine<'_, B>, clients: &mut HashMap<RequestId, Client>, s: Submission| {
            metrics
                .prompt_tokens
                .fetch_add(s.prepared.prompt().len() as u64, Ordering::Relaxed);
            let id = engine.add_prepared(s.prepared);
            clients.insert(
                id,
                Client {
                    index: s.index,
                    out: s.out,
                },
            );
        };
    loop {
        if engine.is_idle() {
            // Nothing to compute: wait for a request.
            match received.recv() {
                Ok(s) => admit(engine, &mut clients, s),
                Err(_) => return,
            }
        }
        while let Ok(s) = received.try_recv() {
            admit(engine, &mut clients, s);
        }
        // The answers no one reads any more.
        clients.retain(|&id, client| {
            if client.out.is_closed() {
                engine.cancel(id);
                in_flight.fetch_sub(1, Ordering::Relaxed);
                false
            } else {
                true
            }
        });
        match engine.step() {
            Ok(events) => {
                for event in events {
                    let Some(client) = clients.get(&event.id) else {
                        continue;
                    };
                    if let Some(piece) = event.piece {
                        metrics.generated_tokens.fetch_add(1, Ordering::Relaxed);
                        let _ = client.out.send((client.index, Output::Piece(piece)));
                    }
                    if let Some(finish) = event.finish {
                        let client = clients.remove(&event.id).expect("found above");
                        let _ = client.out.send((
                            client.index,
                            Output::Done {
                                finish,
                                cached: event.shared,
                            },
                        ));
                        in_flight.fetch_sub(1, Ordering::Relaxed);
                    }
                }
            }
            Err(e) => {
                let message = e.error.to_string();
                if is_device_error(&e.error) && !device_answers(pipeline) {
                    // A device that failed for good: say so to the health
                    // checks, which will restart the server.
                    healthy.store(false, Ordering::Relaxed);
                }
                let mut fields = serde_json::Map::new();
                fields.insert("error".into(), message.clone().into());
                fields.insert("requests".into(), e.failed.len().into());
                log::event("pass_failed", fields);
                for id in e.failed {
                    if let Some(client) = clients.remove(&id) {
                        let _ = client
                            .out
                            .send((client.index, Output::Failed(message.clone())));
                        in_flight.fetch_sub(1, Ordering::Relaxed);
                    }
                }
            }
        }
        metrics.engine(&engine.stats());
    }
}
