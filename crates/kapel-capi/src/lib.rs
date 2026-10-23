//! The C API of Kapel: `libkapel` and its header, `include/kapel.h`.
//!
//! The header is the reference: each function here implements the one it
//! declares, with the same name, and each structure mirrors its own, field
//! for field. The work is done by [`kapel_runtime::Model`]; this crate only
//! checks the pointers, converts the strings, and keeps panics from
//! crossing into C ([`std::panic::catch_unwind`]).
//!
//! The option structures start with their size. A structure from a program
//! built against an older header is shorter: the fields it lacks keep
//! their defaults.
//!
//! The CUDA backend comes with the feature `cuda`.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Duration;

use kapel_runtime::{Finish, GenerateOptions, LoadOptions, Model, ServeOptions};

/// `KAPEL_API_VERSION` of the header.
pub const API_VERSION: u32 = 1;

/// What a call returned (`kapel_status`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// `KAPEL_OK`.
    Ok = 0,
    /// `KAPEL_INVALID_ARGUMENT`: a null pointer, text that is not UTF-8, an
    /// unknown option.
    InvalidArgument = 1,
    /// `KAPEL_ERROR`: the call failed.
    Error = 2,
    /// `KAPEL_PANIC`: a bug in Kapel, caught.
    Panic = 3,
}

/// Why a generation ended (`kapel_finish`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// `KAPEL_FINISH_EOS`.
    Eos = 0,
    /// `KAPEL_FINISH_STOP`.
    Stop = 1,
    /// `KAPEL_FINISH_LENGTH`.
    Length = 2,
    /// `KAPEL_FINISH_CANCELLED`.
    Cancelled = 3,
}

/// `kapel_load_options`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CLoadOptions {
    /// Its size in bytes.
    pub size: usize,
    /// `cpu`, `cuda` or `cuda:N`; null for `cpu`.
    pub device: *const c_char,
    /// `f32`, `bf16` or `f16`; null for the device's default.
    pub dtype: *const c_char,
    /// The revision of a repository of the Hub; null for `main`.
    pub revision: *const c_char,
    /// The quantized format of the linear projections; null for none.
    pub quantize: *const c_char,
    /// Nonzero to quantize the output table too.
    pub quantize_output: c_int,
    /// The type of the cache of keys and values; null for that of the keys.
    pub kv_cache: *const c_char,
    /// Nonzero for activations in bytes on the processor.
    pub byte_activations: c_int,
}

/// `kapel_generate_options`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CGenerateOptions {
    /// Its size in bytes.
    pub size: usize,
    /// The most tokens to generate.
    pub max_tokens: usize,
    /// The divisor of the logits; 0 for greedy decoding.
    pub temperature: f64,
    /// Only the `k` most likely tokens; 0 for all.
    pub top_k: usize,
    /// Only the most likely tokens whose probabilities add up to `top_p`.
    pub top_p: f64,
    /// Only the tokens at least `min_p` times as likely as the most likely.
    pub min_p: f64,
    /// The penalty of the tokens already in the sequence; 1 for none.
    pub repetition_penalty: f64,
    /// The seed of the draws.
    pub seed: u64,
    /// `stop_count` texts that end the answer.
    pub stop: *const *const c_char,
    /// The number of `stop`.
    pub stop_count: usize,
}

/// `kapel_message`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CMessage {
    /// `system`, `user`, `assistant`…
    pub role: *const c_char,
    /// Its text.
    pub content: *const c_char,
}

/// `kapel_generation`.
#[repr(C)]
#[derive(Debug)]
pub struct CGeneration {
    /// The whole answer, for [`kapel_string_free`].
    pub text: *mut c_char,
    /// Why it ended.
    pub finish: FinishReason,
    /// The tokens of the prompt.
    pub prompt_tokens: usize,
    /// The tokens generated.
    pub completion_tokens: usize,
}

/// `kapel_serve_options`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CServeOptions {
    /// Its size in bytes.
    pub size: usize,
    /// The address to listen on.
    pub host: *const c_char,
    /// The port.
    pub port: u16,
    /// The model's name in the API; null for one from the name given.
    pub served_model_name: *const c_char,
    /// The most positions of a sequence; 0 for the model's.
    pub max_model_len: usize,
    /// The memory of the cache in GiB; 0 for the default.
    pub cache_memory: f64,
    /// On a card, the share of its memory to use.
    pub gpu_memory_utilization: f64,
    /// The most answers computed at once.
    pub max_sequences: usize,
    /// The most tokens of one pass.
    pub max_pass_tokens: usize,
    /// The most answers in flight.
    pub max_requests: usize,
    /// Nonzero to share the cache of common prompt beginnings.
    pub prefix_sharing: c_int,
    /// The longest a request may take, in seconds.
    pub timeout: u64,
}

/// `kapel_piece_callback`: a piece of text, its length, the user's data;
/// nonzero to stop.
pub type PieceCallback =
    Option<unsafe extern "C" fn(piece: *const c_char, len: usize, user_data: *mut c_void) -> c_int>;

/// `kapel_model`: what [`kapel_model_load`] gives.
pub struct CModel(Model);

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// Why a call failed.
struct Failure(Status, String);

impl Failure {
    fn argument(message: impl Into<String>) -> Self {
        Self(Status::InvalidArgument, message.into())
    }
}

impl From<kapel_runtime::Error> for Failure {
    fn from(e: kapel_runtime::Error) -> Self {
        Self(Status::Error, e.to_string())
    }
}

fn set_error(message: &str) {
    // A NUL inside the message would cut it: it becomes a space.
    let message = CString::new(message.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = message);
}

/// Runs `f`, turning its failure or its panic into a status and the last
/// error.
fn guard(f: impl FnOnce() -> Result<(), Failure>) -> Status {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => {
            set_error("");
            Status::Ok
        }
        Ok(Err(Failure(status, message))) => {
            set_error(&message);
            status
        }
        Err(panic) => {
            let message = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown".into());
            set_error(&format!("Kapel panicked: {message}"));
            Status::Panic
        }
    }
}

/// The string at `ptr`, `None` if null.
///
/// # Safety
///
/// `ptr` is null or points to a NUL-terminated string.
unsafe fn optional_str<'a>(ptr: *const c_char, name: &str) -> Result<Option<&'a str>, Failure> {
    if ptr.is_null() {
        return Ok(None);
    }
    // SAFETY: the caller promises a NUL-terminated string.
    let s = unsafe { CStr::from_ptr(ptr) };
    s.to_str()
        .map(Some)
        .map_err(|_| Failure::argument(format!("{name} is not UTF-8")))
}

/// The string at `ptr`, which must not be null.
///
/// # Safety
///
/// As [`optional_str`].
unsafe fn required_str<'a>(ptr: *const c_char, name: &str) -> Result<&'a str, Failure> {
    // SAFETY: as the caller promises.
    unsafe { optional_str(ptr, name) }?.ok_or_else(|| Failure::argument(format!("{name} is null")))
}

/// The model at `ptr`, which must not be null.
///
/// # Safety
///
/// `ptr` is null or a model from [`kapel_model_load`], not freed.
unsafe fn model<'a>(ptr: *const CModel) -> Result<&'a Model, Failure> {
    // SAFETY: as the caller promises.
    unsafe { ptr.as_ref() }
        .map(|m| &m.0)
        .ok_or_else(|| Failure::argument("model is null"))
}

/// The options at `ptr` over `defaults`: as many bytes as the caller's
/// structure has (its first field, `size`), at most those of `T`. `None`
/// for a null pointer.
///
/// # Safety
///
/// `ptr` is null, or points to a structure whose first field is its size in
/// bytes, readable over that size, and whose fields are those of `T` in
/// order (an older and shorter `T`).
unsafe fn read_options<T: Copy>(ptr: *const T, defaults: T) -> Result<Option<T>, Failure> {
    if ptr.is_null() {
        return Ok(None);
    }
    // SAFETY: the first field is a `size_t`, readable.
    let size = unsafe { ptr.cast::<usize>().read_unaligned() };
    if size < std::mem::size_of::<usize>() {
        return Err(Failure::argument(
            "an options structure has no size: fill it with its _init function first",
        ));
    }
    let mut value = defaults;
    let len = size.min(std::mem::size_of::<T>());
    // SAFETY: `len` bytes are readable at `ptr` (the caller's size) and
    // writable at `value` (at most its size); the bytes are those of the
    // same fields, and plain data.
    unsafe {
        std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), (&raw mut value).cast::<u8>(), len);
    }
    Ok(Some(value))
}

/// Writes `ids` to the caller's pointers, for [`kapel_tokens_free`].
///
/// # Safety
///
/// `ids` and `len` are writable, or null (an error then).
unsafe fn give_tokens(
    tokens: Vec<u32>,
    ids: *mut *mut u32,
    len: *mut usize,
) -> Result<(), Failure> {
    if ids.is_null() || len.is_null() {
        return Err(Failure::argument("ids or len is null"));
    }
    let tokens = Box::into_raw(tokens.into_boxed_slice());
    // SAFETY: both are writable, as the caller promises.
    unsafe {
        len.write(tokens.len());
        ids.write(tokens.cast::<u32>());
    }
    Ok(())
}

/// A string for the caller, for [`kapel_string_free`].
fn give_string(text: String) -> *mut c_char {
    CString::new(text.replace('\0', " "))
        .unwrap_or_default()
        .into_raw()
}

/// The defaults of `kapel_load_options`.
fn load_defaults() -> CLoadOptions {
    CLoadOptions {
        size: std::mem::size_of::<CLoadOptions>(),
        device: std::ptr::null(),
        dtype: std::ptr::null(),
        revision: std::ptr::null(),
        quantize: std::ptr::null(),
        quantize_output: 0,
        kv_cache: std::ptr::null(),
        byte_activations: 0,
    }
}

/// The options of [`Model::load`] from the caller's.
///
/// # Safety
///
/// `ptr` is null or a `kapel_load_options` whose strings are null or
/// NUL-terminated.
unsafe fn load_options(ptr: *const CLoadOptions) -> Result<LoadOptions, Failure> {
    // SAFETY: as the caller promises.
    let Some(c) = (unsafe { read_options(ptr, load_defaults()) })? else {
        return Ok(LoadOptions::default());
    };
    // SAFETY: the strings are null or NUL-terminated.
    let (device, dtype, quantize, kv_cache, revision) = unsafe {
        (
            optional_str(c.device, "device")?,
            optional_str(c.dtype, "dtype")?,
            optional_str(c.quantize, "quantize")?,
            optional_str(c.kv_cache, "kv_cache")?,
            optional_str(c.revision, "revision")?,
        )
    };
    let mut options = LoadOptions::parse(device, dtype, quantize, kv_cache)
        .map_err(|e| Failure::argument(e.to_string()))?;
    if let Some(revision) = revision {
        options.revision = revision.to_string();
    }
    options.quantize_output = c.quantize_output != 0;
    options.byte_activations = c.byte_activations != 0;
    Ok(options)
}

/// The C structure of `options`.
fn c_generate_options(options: &GenerateOptions) -> CGenerateOptions {
    CGenerateOptions {
        size: std::mem::size_of::<CGenerateOptions>(),
        max_tokens: options.max_tokens,
        temperature: options.temperature,
        top_k: options.top_k.unwrap_or(0),
        top_p: options.top_p,
        min_p: options.min_p,
        repetition_penalty: options.repetition_penalty,
        seed: options.seed,
        stop: std::ptr::null(),
        stop_count: 0,
    }
}

/// `KAPEL_API_VERSION`, as the library was built.
#[unsafe(no_mangle)]
pub extern "C" fn kapel_api_version() -> u32 {
    API_VERSION
}

/// The version of Kapel.
#[unsafe(no_mangle)]
pub extern "C" fn kapel_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// The message of the last failure on this thread, or `""`.
#[unsafe(no_mangle)]
pub extern "C" fn kapel_last_error() -> *const c_char {
    // The string lives in the thread's storage until the next call sets it.
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Fills `options` with the defaults.
///
/// # Safety
///
/// `options` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kapel_load_options_init(options: *mut CLoadOptions) {
    if !options.is_null() {
        // SAFETY: writable, as the caller promises.
        unsafe { options.write(load_defaults()) };
    }
}

/// Loads `model` into `*out`.
///
/// # Safety
///
/// `model` is a NUL-terminated string; `options` is null or initialized
/// options; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kapel_model_load(
    model: *const c_char,
    options: *const CLoadOptions,
    out: *mut *mut CModel,
) -> Status {
    guard(|| {
        if out.is_null() {
            return Err(Failure::argument("out is null"));
        }
        // SAFETY: as the caller promises.
        let (name, options) = unsafe { (required_str(model, "model")?, load_options(options)?) };
        let loaded = Model::load(name, &options)?;
        // SAFETY: writable, as the caller promises.
        unsafe { out.write(Box::into_raw(Box::new(CModel(loaded)))) };
        Ok(())
    })
}

/// Frees a model.
///
/// # Safety
///
/// `model` is null or from [`kapel_model_load`], not freed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kapel_model_free(model: *mut CModel) {
    if !model.is_null() {
        // A panic in a destructor would cross into C: it is dropped here.
        // SAFETY: from `Box::into_raw` in `kapel_model_load`.
        let model = unsafe { Box::from_raw(model) };
        let _ = catch_unwind(AssertUnwindSafe(move || drop(model)));
    }
}

/// The tokens of `text`.
///
/// # Safety
///
/// `model` is a loaded model, `text` a NUL-terminated string, `ids` and
/// `len` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kapel_tokenize(
    model: *const CModel,
    text: *const c_char,
    add_special_tokens: c_int,
    ids: *mut *mut u32,
    len: *mut usize,
) -> Status {
    guard(|| {
        // SAFETY: as the caller promises.
        let (model, text) = unsafe { (self::model(model)?, required_str(text, "text")?) };
        let tokens = model.tokenize(text, add_special_tokens != 0)?;
        // SAFETY: as the caller promises.
        unsafe { give_tokens(tokens, ids, len) }
    })
}

/// The text of `len` tokens.
///
/// # Safety
///
/// `model` is a loaded model, `ids` points to `len` tokens (or is null with
/// `len` 0), `text` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kapel_detokenize(
    model: *const CModel,
    ids: *const u32,
    len: usize,
    skip_special_tokens: c_int,
    text: *mut *mut c_char,
) -> Status {
    guard(|| {
        // SAFETY: as the caller promises.
        let model = unsafe { self::model(model)? };
        if text.is_null() || (ids.is_null() && len > 0) {
            return Err(Failure::argument("ids or text is null"));
        }
        let ids = if len == 0 {
            &[][..]
        } else {
            // SAFETY: `len` tokens at `ids`, as the caller promises.
            unsafe { std::slice::from_raw_parts(ids, len) }
        };
        let decoded = model.detokenize(ids, skip_special_tokens != 0);
        // SAFETY: writable, as the caller promises.
        unsafe { text.write(give_string(decoded)) };
        Ok(())
    })
}

/// The tokens of a conversation in the model's chat template.
///
/// # Safety
///
/// `model` is a loaded model, `messages` points to `count` messages of
/// NUL-terminated strings, `ids` and `len` are writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kapel_chat_prompt(
    model: *const CModel,
    messages: *const CMessage,
    count: usize,
    ids: *mut *mut u32,
    len: *mut usize,
) -> Status {
    guard(|| {
        // SAFETY: as the caller promises.
        let model = unsafe { self::model(model)? };
        if messages.is_null() && count > 0 {
            return Err(Failure::argument("messages is null"));
        }
        let messages = if count == 0 {
            &[][..]
        } else {
            // SAFETY: `count` messages, as the caller promises.
            unsafe { std::slice::from_raw_parts(messages, count) }
        };
        let pairs = messages
            .iter()
            .map(|m| {
                // SAFETY: NUL-terminated strings, as the caller promises.
                unsafe {
                    Ok((
                        required_str(m.role, "role")?,
                        required_str(m.content, "content")?,
                    ))
                }
            })
            .collect::<Result<Vec<_>, Failure>>()?;
        let tokens = model.chat_prompt(&pairs)?;
        // SAFETY: as the caller promises.
        unsafe { give_tokens(tokens, ids, len) }
    })
}

/// Fills `options` with the model's sampling settings.
///
/// # Safety
///
/// `model` is a loaded model, `options` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kapel_generate_options_init(
    model: *const CModel,
    options: *mut CGenerateOptions,
) -> Status {
    guard(|| {
        // SAFETY: as the caller promises.
        let model = unsafe { self::model(model)? };
        if options.is_null() {
            return Err(Failure::argument("options is null"));
        }
        let defaults = GenerateOptions::from(model.default_request());
        // SAFETY: writable, as the caller promises.
        unsafe { options.write(c_generate_options(&defaults)) };
        Ok(())
    })
}

/// Generates what follows `prompt`.
///
/// # Safety
///
/// `model` is a loaded model; `prompt` points to `len` tokens; `options` is
/// null or initialized options whose `stop` points to `stop_count`
/// NUL-terminated strings; `callback`, if any, is safe to call with
/// `user_data`; `result` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kapel_generate(
    model: *const CModel,
    prompt: *const u32,
    len: usize,
    options: *const CGenerateOptions,
    callback: PieceCallback,
    user_data: *mut c_void,
    result: *mut CGeneration,
) -> Status {
    guard(|| {
        // SAFETY: as the caller promises.
        let model = unsafe { self::model(model)? };
        if prompt.is_null() || len == 0 {
            return Err(Failure::argument("the prompt is empty"));
        }
        // SAFETY: `len` tokens, as the caller promises.
        let prompt = unsafe { std::slice::from_raw_parts(prompt, len) };
        let defaults = GenerateOptions::from(model.default_request());
        // SAFETY: as the caller promises.
        let c = unsafe { read_options(options, c_generate_options(&defaults))? }
            .unwrap_or_else(|| c_generate_options(&defaults));
        if c.stop.is_null() && c.stop_count > 0 {
            return Err(Failure::argument("stop is null"));
        }
        let stop = (0..c.stop_count)
            .map(|i| {
                // SAFETY: `stop_count` strings at `stop`, as the caller promises.
                unsafe { required_str(*c.stop.add(i), "stop").map(str::to_string) }
            })
            .collect::<Result<Vec<_>, Failure>>()?;
        let options = GenerateOptions {
            max_tokens: c.max_tokens,
            temperature: c.temperature,
            top_k: (c.top_k > 0).then_some(c.top_k),
            top_p: c.top_p,
            min_p: c.min_p,
            repetition_penalty: c.repetition_penalty,
            seed: c.seed,
            stop,
        };
        let outcome = model.generate(prompt, &options, |piece| match callback {
            Some(callback) => {
                let piece = CString::new(piece.replace('\0', " ")).unwrap_or_default();
                let len = piece.as_bytes().len();
                // SAFETY: the caller promises the callback is safe to call
                // with its data; the piece lives through the call.
                unsafe { callback(piece.as_ptr(), len, user_data) == 0 }
            }
            None => true,
        })?;
        if !result.is_null() {
            let finish = match outcome.finish {
                Finish::Eos => FinishReason::Eos,
                Finish::Stop(_) => FinishReason::Stop,
                Finish::Length => FinishReason::Length,
                Finish::Cancelled => FinishReason::Cancelled,
            };
            // SAFETY: writable, as the caller promises.
            unsafe {
                result.write(CGeneration {
                    text: give_string(outcome.text),
                    finish,
                    prompt_tokens: outcome.prompt_tokens,
                    completion_tokens: outcome.completion_tokens,
                });
            }
        }
        Ok(())
    })
}

/// The defaults of `kapel_serve_options`.
fn serve_defaults() -> CServeOptions {
    let d = ServeOptions::default();
    CServeOptions {
        size: std::mem::size_of::<CServeOptions>(),
        host: std::ptr::null(),
        port: d.port,
        served_model_name: std::ptr::null(),
        max_model_len: 0,
        cache_memory: 0.0,
        gpu_memory_utilization: d.gpu_memory_utilization,
        max_sequences: d.max_sequences,
        max_pass_tokens: d.max_pass_tokens,
        max_requests: d.max_requests,
        prefix_sharing: c_int::from(d.prefix_sharing),
        timeout: d.timeout.as_secs(),
    }
}

/// Fills `options` with the defaults of `kapel serve`.
///
/// # Safety
///
/// `options` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kapel_serve_options_init(options: *mut CServeOptions) {
    if !options.is_null() {
        // SAFETY: writable, as the caller promises.
        unsafe { options.write(serve_defaults()) };
    }
}

/// Loads `model` and serves it until SIGINT.
///
/// # Safety
///
/// `model` is a NUL-terminated string; `load` and `options` are null or
/// initialized options.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kapel_serve(
    model: *const c_char,
    load: *const CLoadOptions,
    options: *const CServeOptions,
) -> Status {
    guard(|| {
        // SAFETY: as the caller promises.
        let (name, load) = unsafe { (required_str(model, "model")?, load_options(load)?) };
        // SAFETY: as the caller promises.
        let c = unsafe { read_options(options, serve_defaults())? }.unwrap_or_else(serve_defaults);
        let d = ServeOptions::default();
        // SAFETY: the strings are null or NUL-terminated.
        let (host, served) = unsafe {
            (
                optional_str(c.host, "host")?,
                optional_str(c.served_model_name, "served_model_name")?,
            )
        };
        let options = ServeOptions {
            host: host.map_or(d.host, str::to_string),
            port: c.port,
            name: served.map(str::to_string),
            max_model_len: (c.max_model_len > 0).then_some(c.max_model_len),
            cache_memory: (c.cache_memory > 0.0).then_some(c.cache_memory),
            gpu_memory_utilization: c.gpu_memory_utilization,
            max_sequences: c.max_sequences,
            max_pass_tokens: c.max_pass_tokens,
            max_requests: c.max_requests,
            prefix_sharing: c.prefix_sharing != 0,
            timeout: Duration::from_secs(c.timeout),
        };
        Model::load(name, &load)?.serve(&options)?;
        Ok(())
    })
}

/// Frees tokens from [`kapel_tokenize`] or [`kapel_chat_prompt`].
///
/// # Safety
///
/// `ids` is null, or `len` tokens from one of those calls, not freed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kapel_tokens_free(ids: *mut u32, len: usize) {
    if !ids.is_null() {
        // SAFETY: from `Box::into_raw` of a boxed slice of `len` tokens.
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ids, len)) });
    }
}

/// Frees a string from Kapel.
///
/// # Safety
///
/// `text` is null, or a string from Kapel, not freed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kapel_string_free(text: *mut c_char) {
    if !text.is_null() {
        // SAFETY: from `CString::into_raw`.
        drop(unsafe { CString::from_raw(text) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_null_argument_is_an_error_not_a_crash() {
        let mut out = std::ptr::null_mut();
        // SAFETY: null pointers, which the call refuses.
        let status = unsafe { kapel_model_load(std::ptr::null(), std::ptr::null(), &mut out) };
        assert_eq!(status, Status::InvalidArgument);
        // SAFETY: the string of this thread, read right away.
        let message = unsafe { CStr::from_ptr(kapel_last_error()) };
        assert_eq!(message.to_str().unwrap(), "model is null");
        assert!(out.is_null());
    }

    #[test]
    fn a_bad_option_is_named() {
        let mut options = load_defaults();
        options.device = c"gpu".as_ptr();
        let mut out = std::ptr::null_mut();
        // SAFETY: valid strings and options.
        let status = unsafe { kapel_model_load(c"some/model".as_ptr(), &options, &mut out) };
        assert_eq!(status, Status::InvalidArgument);
        // SAFETY: the string of this thread, read right away.
        let message = unsafe { CStr::from_ptr(kapel_last_error()) };
        assert!(message.to_str().unwrap().contains("gpu"));
    }

    #[test]
    fn a_shorter_structure_keeps_the_defaults_of_its_missing_fields() {
        // A program built when the structure ended after `dtype`.
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Older {
            size: usize,
            device: *const c_char,
            dtype: *const c_char,
        }
        let older = Older {
            size: std::mem::size_of::<Older>(),
            device: c"cpu".as_ptr(),
            dtype: c"bf16".as_ptr(),
        };
        let mut defaults = load_defaults();
        defaults.byte_activations = 7;
        // SAFETY: the older structure has the first fields, in order.
        let read = unsafe { read_options((&raw const older).cast::<CLoadOptions>(), defaults) }
            .ok()
            .flatten()
            .unwrap();
        assert_eq!(read.dtype, older.dtype);
        assert_eq!(read.byte_activations, 7);
        assert!(read.quantize.is_null());
        // A structure never initialized is refused.
        let zero = Older { size: 0, ..older };
        // SAFETY: as above.
        let read = unsafe { read_options((&raw const zero).cast::<CLoadOptions>(), defaults) };
        assert!(read.is_err());
    }

    #[test]
    fn the_version_is_the_crate_s() {
        // SAFETY: a static string.
        let version = unsafe { CStr::from_ptr(kapel_version()) };
        assert_eq!(version.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
        assert_eq!(kapel_api_version(), API_VERSION);
    }
}
