//! The HTTP server: its routes, and how an answer is sent, whole or as a
//! stream of server-sent events.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures_util::stream::{self, Stream};
use kapel_infer::{EngineOptions, Finish, Pipeline, Prepared};
use kapel_ops::{Backend, Transfer};
use kapel_tokenizers::Message;
use serde_json::{Value, json};
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

use crate::api::{self, ApiError, ChatRequest, CompletionRequest, Plan, Prompt, Usage};
use crate::log::{self, fields};
use crate::metrics::{Metrics, Outcome};
use crate::worker::{Output, Submission, Worker};

/// How the server listens and what it accepts.
#[derive(Clone, Debug)]
pub struct ServeOptions {
    /// The address to listen on, such as `127.0.0.1:8000`; port 0 for any
    /// free port.
    pub address: SocketAddr,
    /// The name of the model in the API (`model` of the requests).
    pub model_name: String,
    /// The most positions of one sequence, prompt and answer.
    pub max_model_len: usize,
    /// The most answers waiting or in progress; beyond, requests are refused
    /// with 429.
    pub max_requests: usize,
    /// The most bytes of a request's body.
    pub max_body_bytes: usize,
    /// The longest a request may take, from its arrival to its last token.
    pub request_timeout: Duration,
}

impl ServeOptions {
    /// Defaults for a model called `model_name` of `max_model_len`
    /// positions, on `127.0.0.1:8000`.
    pub fn new(model_name: impl Into<String>, max_model_len: usize) -> Self {
        Self {
            address: SocketAddr::from(([127, 0, 0, 1], 8000)),
            model_name: model_name.into(),
            max_model_len,
            max_requests: 1024,
            max_body_bytes: 4 << 20,
            request_timeout: Duration::from_secs(600),
        }
    }
}

/// What every handler reads.
struct AppState<B> {
    pipeline: Arc<Pipeline<B>>,
    worker: Worker,
    metrics: Arc<Metrics>,
    options: ServeOptions,
    /// When the server started, in seconds since 1970.
    created: u64,
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// A server running on its own threads.
pub struct Server {
    address: SocketAddr,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<std::io::Result<()>>>,
}

impl Server {
    /// Starts serving `pipeline`: its engine on one thread, the HTTP server
    /// on others. Returns once it listens.
    ///
    /// # Errors
    ///
    /// If the address cannot be bound, or the threads cannot start.
    pub fn start<B>(
        pipeline: Pipeline<B>,
        engine: EngineOptions,
        options: ServeOptions,
    ) -> std::io::Result<Self>
    where
        B: Backend + Transfer + Send + Sync + 'static,
    {
        let pipeline = Arc::new(pipeline);
        let metrics = Arc::new(Metrics::new());
        let worker = Worker::start(Arc::clone(&pipeline), engine.clone(), Arc::clone(&metrics));
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("kapel-http")
            .build()?;
        let listener = runtime.block_on(tokio::net::TcpListener::bind(options.address))?;
        let address = listener.local_addr()?;
        log::event(
            "listening",
            fields! {
                "address" => address.to_string(),
                "model" => options.model_name,
                "max_model_len" => options.max_model_len,
                "cache_pages" => engine.pages,
                "prefix_sharing" => engine.prefix_sharing,
            },
        );
        let state = Arc::new(AppState {
            pipeline,
            worker,
            metrics,
            created: now_secs(),
            options,
        });
        let app = router(state);
        let (shutdown, stop) = tokio::sync::oneshot::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("kapel-serve".into())
            .spawn(move || {
                runtime.block_on(async move {
                    axum::serve(listener, app)
                        .with_graceful_shutdown(async move {
                            tokio::select! {
                                _ = stop => {}
                                _ = tokio::signal::ctrl_c() => {}
                            }
                        })
                        .await
                })
            })?;
        Ok(Self {
            address,
            shutdown: Some(shutdown),
            thread: Some(thread),
        })
    }

    /// The address it listens on.
    pub fn address(&self) -> SocketAddr {
        self.address
    }

    /// Waits until the server stops (on Ctrl-C).
    ///
    /// # Errors
    ///
    /// If serving failed.
    pub fn wait(mut self) -> std::io::Result<()> {
        let thread = self.thread.take().expect("a server thread");
        thread
            .join()
            .unwrap_or_else(|_| Err(std::io::Error::other("the server thread panicked")))
    }

    /// Stops the server: requests in progress are answered first.
    ///
    /// # Errors
    ///
    /// If serving failed.
    pub fn stop(mut self) -> std::io::Result<()> {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        self.wait()
    }
}

/// Serves `pipeline` until Ctrl-C.
///
/// # Errors
///
/// If the address cannot be bound, or serving fails.
pub fn serve<B>(
    pipeline: Pipeline<B>,
    engine: EngineOptions,
    options: ServeOptions,
) -> std::io::Result<()>
where
    B: Backend + Transfer + Send + Sync + 'static,
{
    Server::start(pipeline, engine, options)?.wait()
}

fn router<B>(state: Arc<AppState<B>>) -> Router
where
    B: Backend + Transfer + Send + Sync + 'static,
{
    let limit = state.options.max_body_bytes;
    Router::new()
        .route("/v1/chat/completions", post(chat::<B>))
        .route("/v1/completions", post(completions::<B>))
        .route("/v1/models", get(models::<B>))
        .route("/health", get(health::<B>))
        .route("/metrics", get(metrics::<B>))
        .fallback(|| async {
            error_response(&ApiError {
                status: 404,
                kind: "not_found_error",
                message: "no such route".into(),
                param: None,
            })
        })
        .layer(DefaultBodyLimit::max(limit))
        .with_state(state)
}

fn error_response(error: &ApiError) -> Response {
    let status = StatusCode::from_u16(error.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, axum::Json(error.body())).into_response()
}

async fn models<B>(State(state): State<Arc<AppState<B>>>) -> Response {
    axum::Json(json!({
        "object": "list",
        "data": [{
            "id": state.options.model_name,
            "object": "model",
            "created": state.created,
            "owned_by": "kapel",
            "max_model_len": state.options.max_model_len,
        }]
    }))
    .into_response()
}

async fn health<B>(State(state): State<Arc<AppState<B>>>) -> Response {
    if state.worker.is_alive() {
        axum::Json(json!({ "status": "ok" })).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(json!({ "status": "the engine stopped, or its device failed" })),
        )
            .into_response()
    }
}

async fn metrics<B>(State(state): State<Arc<AppState<B>>>) -> Response {
    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4"),
        )],
        state.metrics.render(),
    )
        .into_response()
}

/// Which endpoint a request came to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Chat,
    Completion,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Completion => "completion",
        }
    }
}

/// A request read and tokenized: what to submit.
struct Ready {
    plan: Plan,
    /// Per answer, ready for the engine.
    prepared: Vec<Prepared>,
    /// Per answer, the tokens of its prompt.
    prompt_tokens: Vec<usize>,
}

fn read_json<T: serde::de::DeserializeOwned>(body: &Bytes) -> Result<T, ApiError> {
    serde_json::from_slice(body)
        .map_err(|e| ApiError::invalid(format!("the request is not valid: {e}"), None))
}

fn check_model<B>(state: &AppState<B>, model: Option<&String>) -> Result<(), ApiError> {
    match model {
        Some(name) if name != &state.options.model_name => Err(ApiError {
            status: 404,
            kind: "not_found_error",
            message: format!(
                "the model {name:?} does not exist; this server serves {:?}",
                state.options.model_name
            ),
            param: Some("model"),
        }),
        _ => Ok(()),
    }
}

/// The answers of `prompts` (each `plan.choices` times), each with its
/// length: what the request asks, or else the rest of the context.
fn prepare<B: Backend + Transfer>(
    state: &AppState<B>,
    plan: Plan,
    prompts: Vec<Vec<u32>>,
    default_tokens: Option<usize>,
) -> Result<Ready, ApiError> {
    let max_len = state.options.max_model_len;
    let mut prepared = Vec::new();
    let mut prompt_tokens = Vec::new();
    for prompt in prompts {
        if prompt.is_empty() {
            return Err(ApiError::invalid("the prompt is empty", Some("prompt")));
        }
        if prompt.len() >= max_len {
            return Err(ApiError::invalid(
                format!(
                    "the prompt has {} tokens; this model takes at most {max_len} positions",
                    prompt.len()
                ),
                Some("messages"),
            ));
        }
        let room = max_len - prompt.len();
        let wanted = plan.max_tokens.or(default_tokens).unwrap_or(room);
        if wanted > room {
            return Err(ApiError::invalid(
                format!(
                    "{} prompt tokens and {wanted} to generate exceed the {max_len} positions \
                     of this model",
                    prompt.len()
                ),
                Some("max_tokens"),
            ));
        }
        for choice in 0..plan.choices {
            let mut request = plan.request.clone();
            request.max_tokens = wanted;
            request.sampling.seed = request.sampling.seed.wrapping_add(choice as u64);
            let ready = state
                .pipeline
                .prepare(prompt.clone(), &request)
                .map_err(|e| ApiError::invalid(e.to_string(), Some("response_format")))?;
            prepared.push(ready);
            prompt_tokens.push(prompt.len());
        }
    }
    Ok(Ready {
        plan,
        prepared,
        prompt_tokens,
    })
}

async fn chat<B>(State(state): State<Arc<AppState<B>>>, body: Bytes) -> Response
where
    B: Backend + Transfer + Send + Sync + 'static,
{
    let arrival = Instant::now();
    let read = {
        let state = Arc::clone(&state);
        tokio::task::spawn_blocking(move || -> Result<Ready, ApiError> {
            let request: ChatRequest = read_json(&body)?;
            check_model(&state, request.generation.model.as_ref())?;
            let plan = request
                .generation
                .plan(state.pipeline.default_sampling(), true)?;
            if request.messages.is_empty() {
                return Err(ApiError::invalid("no message", Some("messages")));
            }
            let messages = request
                .messages
                .iter()
                .map(|m| Ok(Message::new(m.role.clone(), m.text()?)))
                .collect::<Result<Vec<_>, ApiError>>()?;
            let prompt = state
                .pipeline
                .chat_prompt(&messages)
                .map_err(|e| ApiError::invalid(e.to_string(), Some("messages")))?;
            prepare(&state, plan, vec![prompt], None)
        })
        .await
    };
    answer(state, Kind::Chat, arrival, read).await
}

async fn completions<B>(State(state): State<Arc<AppState<B>>>, body: Bytes) -> Response
where
    B: Backend + Transfer + Send + Sync + 'static,
{
    let arrival = Instant::now();
    let read = {
        let state = Arc::clone(&state);
        tokio::task::spawn_blocking(move || -> Result<Ready, ApiError> {
            let request: CompletionRequest = read_json(&body)?;
            check_model(&state, request.generation.model.as_ref())?;
            let plan = request
                .generation
                .plan(state.pipeline.default_sampling(), false)?;
            let encode = |text: &str| {
                state
                    .pipeline
                    .prompt(text)
                    .map_err(|e| ApiError::invalid(e.to_string(), Some("prompt")))
            };
            let prompts = match request.prompt {
                Prompt::Text(text) => vec![encode(&text)?],
                Prompt::Texts(texts) => {
                    texts.iter().map(|t| encode(t)).collect::<Result<_, _>>()?
                }
                Prompt::Tokens(tokens) => vec![tokens],
                Prompt::TokenLists(lists) => lists,
            };
            if prompts.is_empty() || prompts.len() * plan.choices > api::MAX_CHOICES {
                return Err(ApiError::invalid(
                    format!("between 1 and {} answers per request", api::MAX_CHOICES),
                    Some("prompt"),
                ));
            }
            let vocab = state.pipeline.model().config().vocab;
            if prompts.iter().flatten().any(|&t| t as usize >= vocab) {
                return Err(ApiError::invalid(
                    "a token is out of the vocabulary",
                    Some("prompt"),
                ));
            }
            // OpenAI's default for this endpoint.
            prepare(&state, plan, prompts, Some(16))
        })
        .await
    };
    answer(state, Kind::Completion, arrival, read).await
}

/// What a request has received so far, for its answer, its metrics and its
/// log line; on drop (the answer sent, or the client gone), it is counted.
struct Tracker<B> {
    state: Arc<AppState<B>>,
    kind: Kind,
    id: String,
    arrival: Instant,
    first: Option<Instant>,
    /// Per answer, its last token's arrival.
    last: Vec<Option<Instant>>,
    texts: Vec<String>,
    finishes: Vec<Option<Finish>>,
    usage: Usage,
    outcome: Outcome,
    error: Option<String>,
}

impl<B> Tracker<B> {
    fn new(state: Arc<AppState<B>>, kind: Kind, arrival: Instant, ready: &Ready) -> Self {
        let n = ready.prepared.len();
        let prefix = match kind {
            Kind::Chat => "chatcmpl",
            Kind::Completion => "cmpl",
        };
        let mut usage = Usage::default();
        for &p in &ready.prompt_tokens {
            usage.add(p, 0, 0);
        }
        Self {
            id: format!("{prefix}-{:016x}", api_id()),
            state,
            kind,
            arrival,
            first: None,
            last: vec![None; n],
            texts: vec![String::new(); n],
            finishes: vec![None; n],
            usage,
            outcome: Outcome::Cancelled,
            error: None,
        }
    }

    /// Notes an output; the text it adds to its answer, if any.
    fn note(&mut self, index: usize, output: Output) -> Option<String> {
        let now = Instant::now();
        match output {
            Output::Piece(piece) => {
                let metrics = &self.state.metrics;
                if self.first.is_none() {
                    self.first = Some(now);
                    metrics
                        .first_token
                        .observe((now - self.arrival).as_secs_f64());
                }
                if let Some(last) = self.last[index] {
                    metrics.between_tokens.observe((now - last).as_secs_f64());
                }
                self.last[index] = Some(now);
                self.usage.add(0, 1, 0);
                self.texts[index].push_str(&piece.text);
                Some(piece.text)
            }
            Output::Done { finish, cached } => {
                if let Finish::Error(e) = &finish {
                    self.error = Some(e.clone());
                }
                self.usage.add(0, 0, cached);
                self.finishes[index] = Some(finish);
                None
            }
            Output::Failed(e) => {
                self.error = Some(e);
                self.finishes[index] = Some(Finish::Error(String::new()));
                None
            }
        }
    }

    fn done(&self) -> bool {
        self.finishes.iter().all(Option::is_some)
    }
}

impl<B> Drop for Tracker<B> {
    fn drop(&mut self) {
        let outcome = match (&self.error, self.done()) {
            (Some(_), _) => Outcome::Failed,
            (None, true) => Outcome::Ok,
            (None, false) => self.outcome,
        };
        let metrics = &self.state.metrics;
        metrics.ended(outcome);
        let duration = self.arrival.elapsed().as_secs_f64();
        if matches!(outcome, Outcome::Ok) {
            metrics.duration.observe(duration);
        }
        let finishes: Vec<&str> = self
            .finishes
            .iter()
            .map(|f| f.as_ref().map_or("none", api::finish_reason))
            .collect();
        log::event(
            "request",
            fields! {
                "id" => self.id,
                "endpoint" => self.kind.name(),
                "outcome" => format!("{outcome:?}").to_lowercase(),
                "prompt_tokens" => self.usage.prompt_tokens,
                "completion_tokens" => self.usage.completion_tokens,
                "cached_tokens" => self.usage.prompt_tokens_details.cached_tokens,
                "first_token_ms" => self.first.map(|f| (f - self.arrival).as_secs_f64() * 1e3),
                "duration_ms" => duration * 1e3,
                "finish" => finishes,
                "error" => self.error,
            },
        );
    }
}

fn api_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    (time << 16) ^ NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Submits a request read by a handler, and answers it.
async fn answer<B>(
    state: Arc<AppState<B>>,
    kind: Kind,
    arrival: Instant,
    read: Result<Result<Ready, ApiError>, tokio::task::JoinError>,
) -> Response
where
    B: Backend + Transfer + Send + Sync + 'static,
{
    let refused = |state: &AppState<B>, error: ApiError| {
        state.metrics.ended(Outcome::Refused);
        log::event(
            "refused",
            fields! {
                "endpoint" => kind.name(),
                "status" => error.status,
                "error" => error.message,
            },
        );
        error_response(&error)
    };
    let ready = match read {
        Ok(Ok(ready)) => ready,
        Ok(Err(error)) => return refused(&state, error),
        Err(e) => {
            return refused(
                &state,
                ApiError {
                    status: 500,
                    kind: "server_error",
                    message: e.to_string(),
                    param: None,
                },
            );
        }
    };
    let count = ready.prepared.len();
    if state.worker.in_flight() + count > state.options.max_requests {
        return refused(
            &state,
            ApiError {
                status: 429,
                kind: "rate_limit_error",
                message: "too many requests are waiting; try again later".into(),
                param: None,
            },
        );
    }
    let mut tracker = Tracker::new(Arc::clone(&state), kind, arrival, &ready);
    let (out, mut received) = unbounded_channel();
    let Ready { plan, prepared, .. } = ready;
    let submissions = prepared
        .into_iter()
        .enumerate()
        .map(|(index, prepared)| Submission {
            prepared,
            index,
            out: out.clone(),
        })
        .collect();
    drop(out);
    if !state.worker.submit(submissions) {
        tracker.outcome = Outcome::Failed;
        tracker.error = Some("the engine has stopped".into());
        return error_response(&ApiError {
            status: 503,
            kind: "server_error",
            message: "the engine has stopped".into(),
            param: None,
        });
    }
    let deadline = tokio::time::Instant::from_std(arrival + state.options.request_timeout);
    if plan.stream {
        let id = HeaderValue::from_str(&tracker.id).ok();
        let events = sse(tracker, received, deadline, plan.include_usage);
        let mut response = Sse::new(events)
            .keep_alive(KeepAlive::default())
            .into_response();
        if let Some(id) = id {
            response.headers_mut().insert("x-request-id", id);
        }
        return response;
    }
    while !tracker.done() {
        match tokio::time::timeout_at(deadline, received.recv()).await {
            Ok(Some((index, output))) => {
                tracker.note(index, output);
            }
            Ok(None) => break,
            Err(_) => {
                tracker.error = Some("timed out".into());
                return error_response(&ApiError {
                    status: 504,
                    kind: "timeout_error",
                    message: "the request took too long".into(),
                    param: None,
                });
            }
        }
    }
    if let Some(error) = &tracker.error {
        return error_response(&ApiError {
            status: 500,
            kind: "server_error",
            message: error.clone(),
            param: None,
        });
    }
    let body = whole(&tracker);
    let mut response = axum::Json(body).into_response();
    if let Ok(id) = HeaderValue::from_str(&tracker.id) {
        response.headers_mut().insert("x-request-id", id);
    }
    response
}

/// The answer of a request that does not stream.
fn whole<B>(t: &Tracker<B>) -> Value {
    let finish = |i: usize| t.finishes[i].as_ref().map(api::finish_reason);
    let choices: Vec<Value> = (0..t.texts.len())
        .map(|i| match t.kind {
            Kind::Chat => json!({
                "index": i,
                "message": { "role": "assistant", "content": t.texts[i] },
                "logprobs": Value::Null,
                "finish_reason": finish(i),
            }),
            Kind::Completion => json!({
                "index": i,
                "text": t.texts[i],
                "logprobs": Value::Null,
                "finish_reason": finish(i),
            }),
        })
        .collect();
    json!({
        "id": t.id,
        "object": match t.kind { Kind::Chat => "chat.completion", Kind::Completion => "text_completion" },
        "created": now_secs(),
        "model": t.state.options.model_name,
        "choices": choices,
        "usage": t.usage,
    })
}

/// One chunk of a streamed answer.
fn chunk<B>(t: &Tracker<B>, choices: Vec<Value>, usage: Option<Value>) -> Value {
    let mut chunk = json!({
        "id": t.id,
        "object": match t.kind { Kind::Chat => "chat.completion.chunk", Kind::Completion => "text_completion" },
        "created": now_secs(),
        "model": t.state.options.model_name,
        "choices": choices,
    });
    if let Some(usage) = usage {
        chunk["usage"] = usage;
    }
    chunk
}

/// What a stream still has to send.
struct Streaming<B> {
    tracker: Tracker<B>,
    received: UnboundedReceiver<(usize, Output)>,
    deadline: tokio::time::Instant,
    include_usage: bool,
    /// Per answer, whether its first chunk (with the role) went.
    started: Vec<bool>,
    /// Chunks ready to go before reading more.
    queue: std::collections::VecDeque<SseEvent>,
    ended: bool,
}

fn data(value: &Value) -> SseEvent {
    SseEvent::default().data(value.to_string())
}

/// The answer as server-sent events: one chunk per piece of text, then the
/// end of each answer, the usage if asked, and `[DONE]`.
fn sse<B: Send + Sync + 'static>(
    tracker: Tracker<B>,
    received: UnboundedReceiver<(usize, Output)>,
    deadline: tokio::time::Instant,
    include_usage: bool,
) -> impl Stream<Item = Result<SseEvent, std::convert::Infallible>> + Send {
    let n = tracker.texts.len();
    let start = Streaming {
        tracker,
        received,
        deadline,
        include_usage,
        started: vec![false; n],
        queue: std::collections::VecDeque::new(),
        ended: false,
    };
    stream::unfold(start, |mut s| async move {
        loop {
            if let Some(event) = s.queue.pop_front() {
                return Some((Ok(event), s));
            }
            if s.ended {
                return None;
            }
            if s.tracker.done() {
                if s.include_usage && s.tracker.error.is_none() {
                    let usage = serde_json::to_value(s.tracker.usage).expect("serializes");
                    let last = chunk(&s.tracker, Vec::new(), Some(usage));
                    s.queue.push_back(data(&last));
                }
                s.queue.push_back(SseEvent::default().data("[DONE]"));
                s.ended = true;
                continue;
            }
            let (index, output) = match tokio::time::timeout_at(s.deadline, s.received.recv()).await
            {
                Ok(Some(next)) => next,
                Ok(None) => return None,
                Err(_) => {
                    s.tracker.error = Some("timed out".into());
                    let error = ApiError {
                        status: 504,
                        kind: "timeout_error",
                        message: "the request took too long".into(),
                        param: None,
                    };
                    s.queue.push_back(data(&error.body()));
                    s.queue.push_back(SseEvent::default().data("[DONE]"));
                    s.ended = true;
                    continue;
                }
            };
            let kind = s.tracker.kind;
            let usage_field = s.include_usage.then_some(Value::Null);
            match s.tracker.note(index, output) {
                Some(text) => {
                    let first = !s.started[index];
                    s.started[index] = true;
                    if text.is_empty() && !first {
                        continue;
                    }
                    let choice = match kind {
                        Kind::Chat => {
                            let delta = if first {
                                json!({ "role": "assistant", "content": text })
                            } else {
                                json!({ "content": text })
                            };
                            json!({ "index": index, "delta": delta, "logprobs": Value::Null, "finish_reason": Value::Null })
                        }
                        Kind::Completion => {
                            json!({ "index": index, "text": text, "logprobs": Value::Null, "finish_reason": Value::Null })
                        }
                    };
                    let c = chunk(&s.tracker, vec![choice], usage_field);
                    s.queue.push_back(data(&c));
                }
                None => {
                    if let Some(error) = s.tracker.error.clone() {
                        let error = ApiError {
                            status: 500,
                            kind: "server_error",
                            message: error,
                            param: None,
                        };
                        s.queue.push_back(data(&error.body()));
                        s.queue.push_back(SseEvent::default().data("[DONE]"));
                        s.ended = true;
                        continue;
                    }
                    let reason = s.tracker.finishes[index].as_ref().map(api::finish_reason);
                    let choice = match kind {
                        Kind::Chat => {
                            let delta = if s.started[index] {
                                json!({})
                            } else {
                                json!({ "role": "assistant", "content": "" })
                            };
                            json!({ "index": index, "delta": delta, "logprobs": Value::Null, "finish_reason": reason })
                        }
                        Kind::Completion => {
                            json!({ "index": index, "text": "", "logprobs": Value::Null, "finish_reason": reason })
                        }
                    };
                    s.started[index] = true;
                    let c = chunk(&s.tracker, vec![choice], usage_field);
                    s.queue.push_back(data(&c));
                }
            }
        }
    })
}
