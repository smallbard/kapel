//! The engine of a server, on the processor: requests that come and go,
//! suspended and taken up again, sharing their beginnings, held to a form,
//! give what each would alone. The greedy tokens of SmolLM2-135M saved by
//! `tools/parity/generation_reference.py`; without them, the tests are
//! skipped and say so on stderr, or fail with `KAPEL_REQUIRE_PARITY=1`.

use std::collections::HashMap;
use std::path::PathBuf;

use kapel_core::DType;
use kapel_infer::{Constraint, Engine, EngineOptions, Finish, Pipeline, Request, RequestId};
use kapel_models::PAGE;
use serde_json::Value;

fn reference() -> Option<Value> {
    let file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/parity/generation/smollm2-135m.json");
    let found = std::fs::read(&file)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .filter(|r| PathBuf::from(r["directory"].as_str().unwrap_or_default()).is_dir());
    if found.is_none() {
        assert!(
            std::env::var_os("KAPEL_REQUIRE_PARITY").is_none_or(|v| v != "1"),
            "{} is missing, and KAPEL_REQUIRE_PARITY=1",
            file.display()
        );
        eprintln!(
            "skipped: {} missing (see tools/parity/README.md)",
            file.display()
        );
    }
    found
}

fn ids(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

fn pipeline(reference: &Value) -> Pipeline<kapel_cpu::Cpu> {
    let dir = reference["directory"].as_str().unwrap();
    Pipeline::load(dir, kapel_cpu::Cpu::new(), Some(DType::F32)).unwrap()
}

/// Steps until the engine is idle; the tokens and the end of each request.
fn run(
    engine: &mut Engine<'_, kapel_cpu::Cpu>,
    tokens: &mut HashMap<RequestId, (Vec<u32>, Option<Finish>)>,
) {
    while !engine.is_idle() {
        for event in engine.step().unwrap() {
            let entry = tokens.entry(event.id).or_default();
            entry.0.push(event.piece.unwrap().token);
            if let Some(finish) = event.finish {
                entry.1 = Some(finish);
            }
        }
    }
}

#[test]
fn requests_that_come_and_go_get_their_greedy_tokens() {
    let Some(reference) = reference() else { return };
    let pipeline = pipeline(&reference);
    let cases: Vec<(Vec<u32>, Vec<u32>)> = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| (ids(&c["ids"]), ids(&c["f32"])))
        .collect();
    // Room for the longest request and a little more: the others are
    // suspended when they all grow.
    let longest = cases
        .iter()
        .map(|(p, w)| (p.len() + w.len()).div_ceil(PAGE))
        .max()
        .unwrap();
    let mut engine = pipeline.engine(EngineOptions {
        pages: Some(longest + 3),
        prefix_sharing: true,
        // Prompts in pieces of a few tokens.
        max_pass_tokens: 7,
        ..EngineOptions::default()
    });
    let request = |want: &[u32]| Request {
        max_tokens: want.len(),
        ..Request::default()
    };
    let mut wanted = HashMap::new();
    let mut tokens = HashMap::new();
    let (first, rest) = cases.split_at(1);
    for (prompt, want) in first {
        wanted.insert(engine.add(prompt.clone(), &request(want)).unwrap(), want);
    }
    // The others arrive while the first is computed.
    for _ in 0..3 {
        for event in engine.step().unwrap() {
            tokens
                .entry(event.id)
                .or_insert((Vec::new(), None))
                .0
                .push(event.piece.unwrap().token);
        }
    }
    for (prompt, want) in rest.iter().chain(first) {
        wanted.insert(engine.add(prompt.clone(), &request(want)).unwrap(), want);
    }
    run(&mut engine, &mut tokens);
    for (id, want) in &wanted {
        let (got, finish) = &tokens[id];
        assert_eq!(got, *want, "request {}", id.get());
        assert_eq!(finish, &Some(Finish::Length));
    }
    let stats = engine.stats();
    assert!(stats.preemptions > 0, "{stats:?}");
    assert!(stats.shared_positions > 0, "{stats:?}");
    assert_eq!(stats.free_pages, stats.pages);
}

#[test]
fn a_cancelled_request_gives_its_pages_back() {
    let Some(reference) = reference() else { return };
    let pipeline = pipeline(&reference);
    let prompt = ids(&reference["cases"][0]["ids"]);
    let mut engine = pipeline.engine(EngineOptions {
        pages: Some(64),
        ..EngineOptions::default()
    });
    let id = engine.add(prompt, &Request::default()).unwrap();
    engine.step().unwrap();
    engine.step().unwrap();
    assert!(engine.stats().free_pages < 64);
    engine.cancel(id);
    assert!(engine.is_idle());
    assert_eq!(engine.stats().free_pages, 64);
    assert!(engine.step().unwrap().is_empty());
}

/// The text of a request held to `constraint`, and how it ended.
fn constrained(pipeline: &Pipeline<kapel_cpu::Cpu>, constraint: Constraint) -> (String, Finish) {
    let prompt = pipeline.prompt("The answer is").unwrap();
    let mut engine = pipeline.engine(EngineOptions::default());
    let request = Request {
        max_tokens: 64,
        constraint: Some(constraint),
        ..Request::default()
    };
    engine.add(prompt, &request).unwrap();
    let mut text = String::new();
    loop {
        for event in engine.step().unwrap() {
            if let Some(piece) = event.piece {
                text.push_str(&piece.text);
            }
            if let Some(finish) = event.finish {
                return (text, finish);
            }
        }
    }
}

#[test]
fn answers_take_the_form_they_are_given() {
    let Some(reference) = reference() else { return };
    let pipeline = pipeline(&reference);
    let (text, finish) = constrained(
        &pipeline,
        Constraint::Choice(vec!["yes".into(), "no".into(), "maybe not".into()]),
    );
    assert!(
        ["yes", "no", "maybe not"].contains(&text.as_str()),
        "{text:?}"
    );
    assert_eq!(finish, Finish::Eos);
    let (text, _) = constrained(&pipeline, Constraint::Regex("[0-9]{3}-[a-z]{2}".into()));
    assert_eq!(text.len(), 6, "{text:?}");
    assert!(text[..3].bytes().all(|b| b.is_ascii_digit()), "{text:?}");
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "n": { "type": "integer" }, "name": { "type": "string", "maxLength": 8 } },
        "required": ["n", "name"],
        "additionalProperties": false
    });
    let (text, finish) = constrained(&pipeline, Constraint::JsonSchema(schema));
    assert_eq!(finish, Finish::Eos, "{text:?}");
    let value: Value = serde_json::from_str(&text).unwrap();
    assert!(value["n"].is_i64() && value["name"].is_string(), "{value}");
    assert!(
        pipeline
            .check_constraint(&Constraint::Regex("(".into()))
            .is_err()
    );
}
