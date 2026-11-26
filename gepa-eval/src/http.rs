//! HTTP JSON-protocol evaluator (`http`).

use std::collections::BTreeMap;
use std::time::Duration;

use gepa_core::{Candidate, DataId, Split};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::SingleEval;
use crate::command::parse_json_eval_response;

fn default_timeout_secs() -> u64 {
    300
}

fn default_concurrency() -> usize {
    1
}

/// Configuration for a remote or local HTTP evaluation endpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct HttpEvalConfig {
    pub url: String,
    pub headers: BTreeMap<String, String>,
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
}

impl Default for HttpEvalConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            headers: BTreeMap::new(),
            timeout_secs: default_timeout_secs(),
            concurrency: default_concurrency(),
        }
    }
}

fn should_bypass_proxy(url: &str) -> bool {
    let without_scheme = url
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(url);
    let authority = without_scheme.split('/').next().unwrap_or("");
    let host_port = authority.rsplit('@').next().unwrap_or(authority);
    let host = if let Some(rest) = host_port.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        host_port.split(':').next().unwrap_or("")
    };
    if host.eq_ignore_ascii_case("localhost") || host == "::1" {
        return true;
    }
    let mut parts = host.split('.');
    if let (Some(a), Some(b), Some(_), Some(_), None) = (
        parts.next().and_then(|s| s.parse::<u8>().ok()),
        parts.next().and_then(|s| s.parse::<u8>().ok()),
        parts.next().and_then(|s| s.parse::<u8>().ok()),
        parts.next().and_then(|s| s.parse::<u8>().ok()),
        parts.next(),
    ) {
        return a == 127
            || a == 10
            || (a == 172 && (16..=31).contains(&b))
            || (a == 192 && b == 168);
    }
    false
}

pub(crate) fn eval_one(
    cfg: &HttpEvalConfig,
    candidate: &Candidate,
    split: Split,
    id: DataId,
    example: &Value,
    capture_traces: bool,
) -> gepa_core::Result<SingleEval> {
    if cfg.url.trim().is_empty() {
        return Err("http evaluator requires a non-empty `url`".into());
    }

    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(cfg.timeout_secs.max(1)))
        .try_proxy_from_env(!should_bypass_proxy(&cfg.url))
        .build();

    let payload = json!({
        "candidate": candidate,
        "split": split,
        "id": id,
        "example": example,
        "capture_traces": capture_traces,
    });
    let body = serde_json::to_string(&payload)?;

    let mut req = agent.post(&cfg.url).set("Content-Type", "application/json");
    for (k, v) in &cfg.headers {
        req = req.set(k, v);
    }

    let resp = req
        .send_string(&body)
        .map_err(|e| format!("http evaluator request to `{}` failed: {e}", cfg.url))?;
    let text = resp
        .into_string()
        .map_err(|e| format!("failed reading http evaluator response: {e}"))?;
    parse_json_eval_response(&text, capture_traces).map_err(Into::into)
}
