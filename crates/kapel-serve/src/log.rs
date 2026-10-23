//! Structured logs: one JSON object per line on the standard error, with the
//! time (seconds since 1970), the event, and its fields.

use std::io::Write as _;

use serde_json::{Map, Value};

/// Writes one line for `event`.
pub(crate) fn event(event: &str, fields: Map<String, Value>) {
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    let mut line = Map::new();
    line.insert("ts".into(), ((time * 1000.0).round() / 1000.0).into());
    line.insert("event".into(), event.into());
    line.extend(fields);
    let mut text = Value::Object(line).to_string();
    text.push('\n');
    // A full or closed standard error loses the line; serving goes on.
    let _ = std::io::stderr().lock().write_all(text.as_bytes());
}

/// The fields of a log line, from pairs.
macro_rules! fields {
    ($($key:literal => $value:expr),* $(,)?) => {{
        let mut map = serde_json::Map::new();
        $(map.insert($key.into(), serde_json::json!($value));)*
        map
    }};
}

pub(crate) use fields;
