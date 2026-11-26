//! Reflection proposer (`gepa_core::Adapter::propose`): render side info into the
//! instruction-proposal prompt, call an OpenAI-compatible endpoint, and extract the
//! proposed component text.

pub mod client;
pub mod extract;
pub mod template;

use std::collections::BTreeMap;

use gepa_core::{Candidate, DataId, EvalBatch};

pub use client::{LmClient, LmConfig, OpenAiClient, is_loopback_url};
pub use extract::{extract_code_block, strip_thinking};
pub use template::{PromptConfig, format_feedback, render_prompt};

/// Generates mutated candidate component texts using an [`LmClient`] and [`PromptConfig`].
pub struct Reflector<C: LmClient> {
    pub client: C,
    pub prompt: PromptConfig,
}

impl<C: LmClient> Reflector<C> {
    pub fn new(client: C, prompt: PromptConfig) -> Self {
        Self { client, prompt }
    }

    /// Propose updated text for each requested component of `parent` given `eval` on `ids`.
    pub fn propose(
        &mut self,
        parent: &Candidate,
        ids: &[DataId],
        eval: &EvalBatch,
        components: &[String],
    ) -> gepa_core::Result<BTreeMap<String, String>> {
        let mut out = BTreeMap::new();
        for comp in components {
            if !parent.contains_key(comp) {
                return Err(format!("unknown component {comp:?} in candidate").into());
            }
            let prompt = render_prompt(&self.prompt, parent, comp, ids, eval);
            let raw = self.client.complete(&prompt)?;
            let extracted = extract_code_block(&raw);
            out.insert(comp.clone(), extracted);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;

    use serde_json::{Value, json};

    use super::*;

    struct MockLm {
        prompts: Vec<String>,
        reply: String,
    }

    impl LmClient for MockLm {
        fn complete(&mut self, prompt: &str) -> gepa_core::Result<String> {
            self.prompts.push(prompt.to_string());
            Ok(self.reply.clone())
        }
    }

    #[test]
    fn preserves_feedback_key_order_and_hides_keys() {
        let mut obj = serde_json::Map::new();
        obj.insert("Z_first".into(), json!("alpha"));
        obj.insert("Expected answer".into(), json!("42"));
        obj.insert("A_last".into(), json!("beta"));

        let eval = EvalBatch {
            scores: vec![0.25],
            objectives: Some(vec![BTreeMap::from([("acc".into(), 0.25)])]),
            feedback: vec![Value::Object(obj)],
            metric_calls: None,
        };
        let rendered = format_feedback(&[7], &eval, &["Expected answer".into()]);
        let z_pos = rendered.find("Z_first: alpha").expect("Z_first present");
        let a_pos = rendered.find("A_last: beta").expect("A_last present");
        assert!(z_pos < a_pos, "key order should be preserved:\n{rendered}");
        assert!(!rendered.contains("Expected answer"), "hidden key leaked:\n{rendered}");
        assert!(!rendered.contains("42"), "hidden value leaked:\n{rendered}");
        assert!(rendered.contains("acc=0.2500"), "objectives missing:\n{rendered}");
    }

    #[test]
    fn strips_thinking_and_extracts_nested_code_blocks() {
        let resp = "<think>\nHere is a scratchpad:\n```python\nprint(1)\n```\n</think>\nSure, here is the prompt:\n````markdown\n# Rules\n```bash\necho hi\n```\nDone.\n````\n";
        let extracted = extract_code_block(resp);
        assert_eq!(extracted, "# Rules\n```bash\necho hi\n```\nDone.");

        let triple = "<think>hmm</think>\n```text\nfirst\n```\ninner\n```\nlast\n```";
        assert_eq!(extract_code_block(triple), "first\n```\ninner\n```\nlast");

        let unclosed_think = "<think>plain response";
        assert_eq!(extract_code_block(unclosed_think), "<think>plain response");
    }

    #[test]
    fn reflector_propose_uses_template_and_extractor() {
        let mock = MockLm {
            prompts: vec![],
            reply: "<think>analysis</think>\n```\nimproved prompt\n```".into(),
        };
        let cfg = PromptConfig {
            objective: Some("Maximize accuracy".into()),
            background: Some("Math solver".into()),
            hide_keys: vec!["secret".into()],
            ..Default::default()
        };
        let mut reflector = Reflector::new(mock, cfg);
        let parent = Candidate::from([
            ("sys".into(), "old sys".into()),
            ("helper".into(), "helper text".into()),
        ]);
        let mut fb = serde_json::Map::new();
        fb.insert("Problem".into(), json!("1+1"));
        fb.insert("secret".into(), json!("2"));
        let eval = EvalBatch {
            scores: vec![0.0],
            objectives: None,
            feedback: vec![Value::Object(fb)],
            metric_calls: None,
        };
        let next = reflector
            .propose(&parent, &[0], &eval, &["sys".into()])
            .unwrap();
        assert_eq!(next.get("sys").map(String::as_str), Some("improved prompt"));
        let sent = &reflector.client.prompts[0];
        assert!(sent.contains("Maximize accuracy"));
        assert!(sent.contains("Other Components"));
        assert!(sent.contains("helper text"));
        assert!(!sent.contains("secret"));
    }

    fn read_http_request(stream: &mut std::net::TcpStream) {
        let mut buf = Vec::new();
        let mut tmp = [0u8; 1024];
        let mut content_len = None;
        let mut header_end = None;
        loop {
            let n = stream.read(&mut tmp).unwrap_or(0);
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            if header_end.is_none()
                && let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n")
            {
                header_end = Some(pos + 4);
                let headers = String::from_utf8_lossy(&buf[..pos]);
                for line in headers.lines() {
                    if let Some((k, v)) = line.split_once(':')
                        && k.trim().eq_ignore_ascii_case("content-length")
                    {
                        content_len = v.trim().parse::<usize>().ok();
                    }
                }
            }
            if let Some(hend) = header_end {
                let need = hend + content_len.unwrap_or(0);
                if buf.len() >= need {
                    break;
                }
            }
        }
    }

    #[test]
    fn openai_client_retries_500_on_loopback_and_parses_response() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_srv = Arc::clone(&hits);

        let handle = thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                read_http_request(&mut stream);
                let n = hits_srv.fetch_add(1, Ordering::SeqCst);
                let resp = if n == 0 {
                    "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 4\r\nConnection: close\r\n\r\nbusy"
                        .to_string()
                } else {
                    let body = r#"{"choices":[{"message":{"content":"```\nfixed\n```"}}]}"#;
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                };
                stream.write_all(resp.as_bytes()).unwrap();
                let _ = stream.flush();
            }
        });

        assert!(is_loopback_url(&format!("http://{addr}/v1")));
        let mut client = OpenAiClient::new(LmConfig {
            base_url: format!("http://{addr}/v1"),
            model: "qwen-test".into(),
            max_retries: 2,
            timeout_secs: 10,
            ..Default::default()
        });
        let reply = client.complete("hello").unwrap();
        assert_eq!(reply, "```\nfixed\n```");
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        handle.join().unwrap();
    }
}
