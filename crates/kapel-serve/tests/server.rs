//! The server over HTTP, on the processor, with SmolLM2-135M (the weights of
//! `data/parity/generation/smollm2-135m.json`, the chat template of the
//! instruct model in `data/parity/smollm2-instruct`): answers as the
//! pipeline gives them, alone or many at once, whole or streamed; errors as
//! OpenAI gives them; a client that goes away cancels its request. Without
//! the data, the test is skipped, or fails with `KAPEL_REQUIRE_PARITY=1`.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use kapel_core::DType;
use kapel_infer::{EngineOptions, Pipeline, Request};
use kapel_serve::{ServeOptions, Server};
use serde_json::{Value, json};

/// A model directory: the weights of SmolLM2-135M with a chat template.
fn model() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/parity");
    let reference: Option<Value> = std::fs::read(root.join("generation/smollm2-135m.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok());
    let weights = reference
        .as_ref()
        .and_then(|r| r["directory"].as_str())
        .map(PathBuf::from)
        .filter(|d| d.is_dir());
    let template = root.join("smollm2-instruct/chat_template.jinja");
    let (Some(weights), true) = (weights, template.is_file()) else {
        assert!(
            std::env::var_os("KAPEL_REQUIRE_PARITY").is_none_or(|v| v != "1"),
            "the parity data is missing, and KAPEL_REQUIRE_PARITY=1"
        );
        eprintln!("skipped: data/parity missing (see tools/parity/README.md)");
        return None;
    };
    let dir = std::env::temp_dir().join(format!("kapel-serve-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for name in [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
    ] {
        let link = dir.join(name);
        if !link.exists() {
            std::os::unix::fs::symlink(weights.join(name), link).unwrap();
        }
    }
    std::fs::copy(&template, dir.join("chat_template.jinja")).unwrap();
    Some(dir)
}

fn pipeline(dir: &Path) -> Pipeline<kapel_cpu::Cpu> {
    Pipeline::load(dir, kapel_cpu::Cpu::new(), Some(DType::F32)).unwrap()
}

fn send(url: &str, body: &Value) -> Result<ureq::Response, Box<ureq::Error>> {
    ureq::post(url)
        .set("content-type", "application/json")
        .send_string(&body.to_string())
        .map_err(Box::new)
}

fn json_of(response: ureq::Response) -> Value {
    serde_json::from_reader(response.into_reader()).unwrap()
}

fn post(url: &str, body: Value) -> (u16, Value) {
    match send(url, &body).map_err(|e| *e) {
        Ok(r) => (r.status(), json_of(r)),
        Err(ureq::Error::Status(code, r)) => (code, json_of(r)),
        Err(e) => panic!("{e}"),
    }
}

/// The `data:` lines of a stream, `[DONE]` left out, and whether it came.
fn events(url: &str, body: Value) -> (Vec<Value>, bool) {
    let response = send(url, &body).unwrap();
    assert_eq!(response.header("content-type"), Some("text/event-stream"));
    let mut out = Vec::new();
    for line in BufReader::new(response.into_reader()).lines() {
        let line = line.unwrap();
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            return (out, true);
        }
        out.push(serde_json::from_str(data).unwrap());
    }
    (out, false)
}

#[test]
fn the_server_answers_as_openai_does() {
    let Some(dir) = model() else { return };
    // What the pipeline gives alone, greedy.
    let alone = pipeline(&dir);
    let prompt = "The capital of France is";
    let greedy = |text: &str, tokens: usize| -> String {
        let request = Request {
            max_tokens: tokens,
            ..Request::default()
        };
        alone
            .generate(&alone.prompt(text).unwrap(), &request)
            .unwrap()
            .map(|p| p.unwrap().text)
            .collect()
    };
    let want = greedy(prompt, 12);

    let mut options = ServeOptions::new("smollm2", 2048);
    options.address = "127.0.0.1:0".parse().unwrap();
    let engine = EngineOptions {
        pages: Some(512),
        prefix_sharing: true,
        max_pass_tokens: 64,
        ..EngineOptions::default()
    };
    let server = Server::start(pipeline(&dir), engine, options).unwrap();
    let base = format!("http://{}", server.address());
    let url = |path: &str| format!("{base}{path}");

    let models = json_of(ureq::get(&url("/v1/models")).call().unwrap());
    assert_eq!(models["data"][0]["id"], "smollm2");
    assert_eq!(ureq::get(&url("/health")).call().unwrap().status(), 200);

    // A whole answer.
    let (status, body) = post(
        &url("/v1/completions"),
        json!({"model": "smollm2", "prompt": prompt, "max_tokens": 12, "temperature": 0}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["choices"][0]["text"], want.as_str());
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["usage"]["completion_tokens"], 12);

    // The same, streamed, with its usage.
    let (chunks, done) = events(
        &url("/v1/completions"),
        json!({"prompt": prompt, "max_tokens": 12, "temperature": 0, "stream": true,
               "stream_options": {"include_usage": true}}),
    );
    assert!(done);
    let text: String = chunks
        .iter()
        .filter_map(|c| c["choices"].get(0)?.get("text")?.as_str())
        .collect();
    assert_eq!(text, want);
    let usage = &chunks.last().unwrap()["usage"];
    assert_eq!(usage["completion_tokens"], 12);
    // The prompt was seen before: its full pages are shared.
    assert!(
        usage["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            <= 16
    );

    // Many at once, each as alone.
    let prompts = [
        "Once upon a time",
        "The best way to learn",
        "In 1969, the first",
        "Water boils at",
        "My favourite food is",
        "The theory of relativity",
    ];
    let threads: Vec<_> = prompts
        .iter()
        .map(|&p| {
            let url = url("/v1/completions");
            std::thread::spawn(move || {
                post(
                    &url,
                    json!({"prompt": p, "max_tokens": 20, "temperature": 0}),
                )
                .1
            })
        })
        .collect();
    for (p, t) in prompts.iter().zip(threads) {
        let body = t.join().unwrap();
        assert_eq!(body["choices"][0]["text"], greedy(p, 20).as_str(), "{p}");
    }

    // A chat, held to a list of answers, two of them.
    let (status, body) = post(
        &url("/v1/chat/completions"),
        json!({"messages": [{"role": "user", "content": "Is the sky blue?"}],
               "guided_choice": ["yes", "no"], "n": 2, "temperature": 1.0, "seed": 4}),
    );
    assert_eq!(status, 200, "{body}");
    for choice in body["choices"].as_array().unwrap() {
        let content = choice["message"]["content"].as_str().unwrap();
        assert!(["yes", "no"].contains(&content), "{content:?}");
        assert_eq!(choice["finish_reason"], "stop");
    }
    // A chat streamed, its role first.
    let (chunks, done) = events(
        &url("/v1/chat/completions"),
        json!({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5, "stream": true}),
    );
    assert!(done);
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(chunks[0]["object"], "chat.completion.chunk");
    assert_eq!(
        chunks.last().unwrap()["choices"][0]["finish_reason"],
        "length"
    );

    // Errors, as OpenAI gives them.
    for (body, status) in [
        (json!({"model": "other", "prompt": "a"}), 404),
        (json!({"prompt": "a", "logprobs": 2}), 400),
        (json!({"prompt": "a", "max_tokens": 100_000}), 400),
        (json!({"prompt": "a", "guided_regex": "("}), 400),
        (json!({"prompt": 5}), 400),
    ] {
        let (got, error) = post(&url("/v1/completions"), body.clone());
        assert_eq!(got, status, "{body}: {error}");
        assert!(error["error"]["message"].is_string(), "{error}");
    }

    // A client that goes away: its request is cancelled.
    {
        let response = send(
            &url("/v1/completions"),
            &json!({"prompt": prompt, "max_tokens": 1500, "stream": true, "ignore_eos": true}),
        )
        .unwrap();
        let mut reader = BufReader::new(response.into_reader());
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        assert!(line.starts_with("data: "), "{line:?}");
    }
    let metrics = || {
        ureq::get(&url("/metrics"))
            .call()
            .unwrap()
            .into_string()
            .unwrap()
    };
    let start = std::time::Instant::now();
    while !metrics().contains("kapel_requests_total{outcome=\"cancelled\"} 1") {
        assert!(start.elapsed().as_secs() < 30, "{}", metrics());
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    while !metrics().contains("kapel_requests_running 0") {
        assert!(start.elapsed().as_secs() < 30, "{}", metrics());
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let text = metrics();
    assert!(
        text.contains("kapel_time_to_first_token_seconds_count"),
        "{text}"
    );
    server.stop().unwrap();
}
