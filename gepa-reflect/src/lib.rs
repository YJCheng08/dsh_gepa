//! Reflection proposer (`gepa_core::Adapter::propose`): render side info into the
//! instruction-proposal prompt, call an OpenAI-compatible endpoint, and extract the
//! proposed component text.

pub mod client;
pub mod extract;
pub mod lineage;
pub mod template;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use gepa_core::{Candidate, DataId, EvalBatch, Split};

pub use client::{LmClient, LmConfig, OpenAiClient, is_loopback_url, should_bypass_proxy};
pub use extract::{extract_code_block, strip_thinking};
pub use lineage::{EvalRecord, Lineage};
pub use template::{PromptConfig, format_feedback, render_prompt, render_prompt_with_lineage};

/// Generates mutated candidate component texts using an [`LmClient`] and [`PromptConfig`].
pub struct Reflector<C: LmClient> {
    pub client: C,
    pub prompt: PromptConfig,
    pub lineage: Lineage,
}

impl<C: LmClient> Reflector<C> {
    pub fn new(client: C, prompt: PromptConfig) -> Self {
        Self {
            client,
            prompt,
            lineage: Lineage::default(),
        }
    }

    pub fn record_eval(&mut self, candidate: &Candidate, split: Split, eval: &EvalBatch) {
        self.lineage.record_eval(candidate, split, eval);
    }

    /// Propose updated text for each requested component of `parent` given `eval` on `ids`.
    pub fn propose(
        &mut self,
        parent: &Candidate,
        ids: &[DataId],
        eval: &EvalBatch,
        components: &[String],
    ) -> gepa_core::Result<BTreeMap<String, String>> {
        let mut out: BTreeMap<String, String> = BTreeMap::new();
        for comp in components {
            if !parent.contains_key(comp) {
                return Err(format!("unknown component {comp:?} in candidate").into());
            }
            let lineage_ctx = if self.prompt.lineage {
                self.lineage.render_context(parent, comp)
            } else {
                String::new()
            };
            let mut ask_prompt =
                render_prompt_with_lineage(&self.prompt, parent, comp, ids, eval, &lineage_ctx);
            let mut chosen = String::new();
            for attempt in 0..=2 {
                let raw = self.client.complete(&ask_prompt)?;
                let extracted = extract_code_block(&raw);
                chosen = extracted;
                if !self.prompt.lineage || attempt == 2 {
                    break;
                }
                let mut trial = parent.clone();
                trial.extend(out.iter().map(|(k, v)| (k.clone(), v.clone())));
                trial.insert(comp.clone(), chosen.clone());
                if trial == *parent {
                    let _ = write!(
                        ask_prompt,
                        "\n\nYour previous proposal made no changes to `{comp}`; propose a meaningful edit."
                    );
                    continue;
                }
                if let Some(prev) = self.lineage.find(&trial) {
                    let _ = write!(
                        ask_prompt,
                        "\n\nYour previous proposal duplicated an already-evaluated candidate (score={:.4}, status={}); propose a different change.",
                        prev.score, prev.status
                    );
                    continue;
                }
                break;
            }
            out.insert(comp.clone(), chosen);
        }
        let mut final_child = parent.clone();
        final_child.extend(out.iter().map(|(k, v)| (k.clone(), v.clone())));
        self.lineage.note_proposal(parent, &final_child);
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

    #[test]
    fn lineage_renders_failures_cross_branch_and_submetrics() {
        let mut lin = Lineage::default();
        let parent = Candidate::from([("p".into(), "line1\nline2\n".into())]);
        let child_bad = Candidate::from([("p".into(), "line1\nbad_edit\n".into())]);
        let branch_best = Candidate::from([("p".into(), "line1\nbest_edit\n".into())]);
        let sub_lead = Candidate::from([("p".into(), "line1\nfast_edit\n".into())]);

        let eval_with = |s: f64, pass: f64, spd: f64, err: Option<&str>| EvalBatch {
            scores: vec![s],
            objectives: Some(vec![BTreeMap::from([("pass".into(), pass), ("speed".into(), spd)])]),
            feedback: vec![err.map(|e| json!({"error": e})).unwrap_or_else(|| json!({}))],
            metric_calls: None,
        };

        lin.record_eval(&parent, Split::Val, &eval_with(0.4, 0.0, 0.5, None));
        lin.records[0].accepted = true;
        lin.note_proposal(&parent, &child_bad);
        lin.record_eval(&child_bad, Split::Train, &eval_with(0.0, 0.0, 0.0, Some("nvcc compile error")));
        lin.record_eval(&branch_best, Split::Val, &eval_with(0.9, 1.0, 0.8, None));
        lin.records[2].accepted = true;
        lin.record_eval(&sub_lead, Split::Val, &eval_with(0.5, 0.0, 1.9, None));
        lin.records[3].accepted = true;

        let ctx = lin.render_context(&parent, "p");
        assert!(ctx.contains("Trial History & Rejected Edits"), "{ctx}");
        assert!(ctx.contains("nvcc compile error"), "{ctx}");
        assert!(ctx.contains("-line2\n+bad_edit"), "{ctx}");
        assert!(ctx.contains("Best Accepted Branch"), "{ctx}");
        assert!(ctx.contains("+best_edit"), "{ctx}");
        assert!(ctx.contains("Sub-Metric Leaders"), "{ctx}");
        assert!(ctx.contains("same as Best Accepted Branch above"), "{ctx}");
        assert!(ctx.contains("+fast_edit"), "{ctx}");
    }

    #[test]
    fn propose_retries_on_duplicate_in_lineage() {
        struct SeqLm {
            prompts: Vec<String>,
            replies: Vec<String>,
            idx: usize,
        }
        impl LmClient for SeqLm {
            fn complete(&mut self, prompt: &str) -> gepa_core::Result<String> {
                self.prompts.push(prompt.to_string());
                let r = self.replies[self.idx].clone();
                self.idx += 1;
                Ok(r)
            }
        }

        let lm = SeqLm {
            prompts: vec![],
            replies: vec!["```\ndup_text\n```".into(), "```\nfresh_text\n```".into()],
            idx: 0,
        };
        let mut reflector = Reflector::new(lm, PromptConfig::default());
        let parent = Candidate::from([("p".into(), "seed_text".into())]);
        let dup = Candidate::from([("p".into(), "dup_text".into())]);
        reflector.record_eval(&parent, Split::Val, &EvalBatch { scores: vec![0.5], ..Default::default() });
        reflector.record_eval(
            &dup,
            Split::Train,
            &EvalBatch {
                scores: vec![0.0],
                feedback: vec![json!({"error": "bad syntax"})],
                ..Default::default()
            },
        );

        let eval = EvalBatch {
            scores: vec![0.5],
            feedback: vec![json!({"note": "needs work"})],
            ..Default::default()
        };
        let out = reflector.propose(&parent, &[0], &eval, &["p".into()]).unwrap();
        assert_eq!(out["p"], "fresh_text");
        assert_eq!(reflector.client.prompts.len(), 2);
        assert!(reflector.client.prompts[1].contains("duplicated an already-evaluated candidate"));
        assert!(reflector.client.prompts[1].contains("bad syntax"));
    }
}
