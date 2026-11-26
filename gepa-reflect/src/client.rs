//! OpenAI-compatible HTTP reflection client (`ureq` synchronous transport).

use std::thread;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Language model completion abstraction for offline testing and custom backends.
pub trait LmClient {
    fn complete(&mut self, prompt: &str) -> gepa_core::Result<String>;
}

/// Configuration for an OpenAI-compatible `/v1/chat/completions` endpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LmConfig {
    pub model: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub system_prompt: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<usize>,
    pub timeout_secs: u64,
    pub max_retries: usize,
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            model: "default".to_string(),
            base_url: "http://127.0.0.1:8080/v1".to_string(),
            api_key: None,
            api_key_env: None,
            system_prompt: None,
            temperature: Some(0.6),
            max_tokens: None,
            timeout_secs: 300,
            max_retries: 2,
        }
    }
}

/// Returns true if `url` targets a loopback address (`127.*`, `localhost`, or `[::1]`).
pub fn is_loopback_url(url: &str) -> bool {
    let host = extract_host(url);
    host.eq_ignore_ascii_case("localhost") || host == "::1" || is_loopback_v4(host)
}

/// Returns true if `url` targets a loopback or RFC1918 private address that should bypass HTTP proxies.
pub fn should_bypass_proxy(url: &str) -> bool {
    let host = extract_host(url);
    host.eq_ignore_ascii_case("localhost")
        || host == "::1"
        || is_loopback_v4(host)
        || is_rfc1918_v4(host)
}

fn extract_host(url: &str) -> &str {
    let without_scheme = url
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(url);
    let authority = without_scheme.split('/').next().unwrap_or("");
    let host_port = authority.rsplit('@').next().unwrap_or(authority);
    if let Some(rest) = host_port.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        host_port.split(':').next().unwrap_or("")
    }
}

fn parse_ipv4(host: &str) -> Option<[u8; 4]> {
    let mut parts = host.split('.');
    let a = parts.next()?.parse::<u8>().ok()?;
    let b = parts.next()?.parse::<u8>().ok()?;
    let c = parts.next()?.parse::<u8>().ok()?;
    let d = parts.next()?.parse::<u8>().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some([a, b, c, d])
}

fn is_loopback_v4(host: &str) -> bool {
    parse_ipv4(host).is_some_and(|[a, ..]| a == 127)
}

fn is_rfc1918_v4(host: &str) -> bool {
    parse_ipv4(host).is_some_and(|[a, b, ..]| {
        a == 10 || (a == 172 && (16..=31).contains(&b)) || (a == 192 && b == 168)
    })
}

/// Synchronous OpenAI-compatible chat-completions client.
pub struct OpenAiClient {
    config: LmConfig,
    agent: ureq::Agent,
}

impl OpenAiClient {
    pub fn new(config: LmConfig) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(config.timeout_secs.max(1)))
            .try_proxy_from_env(!should_bypass_proxy(&config.base_url))
            .build();
        Self { config, agent }
    }

    fn endpoint(&self) -> String {
        let base = self.config.base_url.trim_end_matches('/');
        if base.ends_with("/chat/completions") {
            base.to_string()
        } else {
            format!("{base}/chat/completions")
        }
    }

    fn resolve_api_key(&self) -> Option<String> {
        if let Some(k) = self.config.api_key.as_ref().filter(|s| !s.is_empty()) {
            return Some(k.clone());
        }
        if let Some(env_name) = self.config.api_key_env.as_ref().filter(|s| !s.is_empty())
            && let Ok(v) = std::env::var(env_name)
            && !v.is_empty()
        {
            return Some(v);
        }
        for name in ["GEPA_LM_API_KEY", "OPENAI_API_KEY"] {
            if let Ok(v) = std::env::var(name)
                && !v.is_empty()
            {
                return Some(v);
            }
        }
        None
    }

    fn build_request_body(&self, prompt: &str) -> String {
        let mut messages = Vec::new();
        if let Some(sys) = self.config.system_prompt.as_ref().filter(|s| !s.is_empty()) {
            messages.push(json!({ "role": "system", "content": sys }));
        }
        messages.push(json!({ "role": "user", "content": prompt }));

        let mut payload = json!({
            "model": self.config.model,
            "messages": messages,
        });
        if let Some(temp) = self.config.temperature {
            payload["temperature"] = json!(temp);
        }
        if let Some(max_tokens) = self.config.max_tokens {
            payload["max_tokens"] = json!(max_tokens);
        }
        serde_json::to_string(&payload).expect("JSON serialization of chat request")
    }

    fn send_once(&self, endpoint: &str, api_key: Option<&str>, body: &str) -> Result<String, SendError> {
        let mut req = self.agent.post(endpoint).set("Content-Type", "application/json");
        if let Some(key) = api_key {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        match req.send_string(body) {
            Ok(resp) => {
                let text = resp.into_string().map_err(|e| SendError::Retryable(e.to_string()))?;
                parse_completion_content(&text).map_err(SendError::Fatal)
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body_snippet = resp.into_string().unwrap_or_default();
                let msg = format!("LM endpoint HTTP {code}: {body_snippet}");
                if code == 429 || code >= 500 {
                    Err(SendError::Retryable(msg))
                } else {
                    Err(SendError::Fatal(msg))
                }
            }
            Err(ureq::Error::Transport(t)) => Err(SendError::Retryable(format!("LM transport error: {t}"))),
        }
    }
}

enum SendError {
    Retryable(String),
    Fatal(String),
}

impl LmClient for OpenAiClient {
    fn complete(&mut self, prompt: &str) -> gepa_core::Result<String> {
        let endpoint = self.endpoint();
        let api_key = self.resolve_api_key();
        let body = self.build_request_body(prompt);
        let mut last_err = String::new();

        for attempt in 0..=self.config.max_retries {
            match self.send_once(&endpoint, api_key.as_deref(), &body) {
                Ok(text) => return Ok(text),
                Err(SendError::Fatal(msg)) => return Err(msg.into()),
                Err(SendError::Retryable(msg)) => {
                    last_err = msg;
                    if attempt < self.config.max_retries {
                        thread::sleep(Duration::from_millis(200 * (attempt as u64 + 1)));
                    }
                }
            }
        }
        Err(last_err.into())
    }
}

fn parse_completion_content(raw_json: &str) -> Result<String, String> {
    let val: Value = serde_json::from_str(raw_json).map_err(|e| format!("invalid JSON from LM: {e}"))?;
    let content = val
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .ok_or_else(|| format!("missing choices[0].message.content in LM response: {raw_json}"))?;

    match content {
        Value::Null => Err("LM returned null content (reasoning may have exhausted max_tokens)".into()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut joined = String::new();
            for part in parts {
                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                    joined.push_str(text);
                }
            }
            if joined.is_empty() {
                Err(format!("empty text parts in choices[0].message.content: {raw_json}"))
            } else {
                Ok(joined)
            }
        }
        other => Err(format!("unexpected content type in LM response: {other}")),
    }
}
