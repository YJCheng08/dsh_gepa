//! Evaluators (`gepa_core::Adapter::evaluate`): command JSON protocol, isolated
//! workspace shell check (`GEPA_SCORE`/`GEPA_SCORES`), and HTTP evaluator with
//! bounded thread concurrency and process-tree timeout kills.

pub mod command;
pub mod http;
pub mod process;
pub mod workspace;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use gepa_core::state::Objectives;
use gepa_core::{Candidate, DataId, EvalBatch, Split};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use command::JsonCommandConfig;
pub use http::HttpEvalConfig;
pub use process::{RunOutput, run_command_with_timeout};
pub use workspace::ShellCheckConfig;

#[derive(Clone, Debug)]
pub(crate) struct SingleEval {
    pub score: f64,
    pub objectives: Option<Objectives>,
    pub feedback: Option<Value>,
    pub metric_calls: Option<usize>,
}

/// Polymorphic evaluator configuration serializable inside `task.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EvalConfig {
    CommandJson(JsonCommandConfig),
    ShellCheck(ShellCheckConfig),
    Http(HttpEvalConfig),
}

impl EvalConfig {
    pub fn concurrency(&self) -> usize {
        match self {
            Self::CommandJson(c) => c.concurrency.max(1),
            Self::ShellCheck(c) => c.concurrency.max(1),
            Self::Http(c) => c.concurrency.max(1),
        }
    }

    pub fn set_concurrency(&mut self, n: usize) {
        let n = n.max(1);
        match self {
            Self::CommandJson(c) => c.concurrency = n,
            Self::ShellCheck(c) => c.concurrency = n,
            Self::Http(c) => c.concurrency = n,
        }
    }
}

/// Host-facing evaluator trait decoupling dataset lookup from evaluation backend.
pub trait Evaluator: Send + Sync {
    fn evaluate(
        &self,
        candidate: &Candidate,
        split: Split,
        ids: &[DataId],
        examples: &[Value],
        capture_traces: bool,
    ) -> gepa_core::Result<EvalBatch>;
}

/// Ready-to-use evaluator constructed from [`EvalConfig`].
#[derive(Clone, Debug)]
pub struct AnyEvaluator {
    pub config: EvalConfig,
}

impl AnyEvaluator {
    pub fn new(config: EvalConfig) -> Self {
        Self { config }
    }

    fn eval_one(
        &self,
        candidate: &Candidate,
        split: Split,
        id: DataId,
        example: &Value,
        capture_traces: bool,
    ) -> gepa_core::Result<SingleEval> {
        match &self.config {
            EvalConfig::CommandJson(cfg) => command::eval_one(cfg, candidate, split, id, example, capture_traces),
            EvalConfig::ShellCheck(cfg) => workspace::eval_one(cfg, candidate, split, id, example, capture_traces),
            EvalConfig::Http(cfg) => http::eval_one(cfg, candidate, split, id, example, capture_traces),
        }
    }
}

impl Evaluator for AnyEvaluator {
    fn evaluate(
        &self,
        candidate: &Candidate,
        split: Split,
        ids: &[DataId],
        examples: &[Value],
        capture_traces: bool,
    ) -> gepa_core::Result<EvalBatch> {
        if ids.len() != examples.len() {
            return Err(format!("evaluator id/example length mismatch: {} vs {}", ids.len(), examples.len()).into());
        }
        if ids.is_empty() {
            return Ok(EvalBatch::default());
        }

        let results = run_parallel(self.config.concurrency(), ids.len(), |i| {
            self.eval_one(candidate, split, ids[i], &examples[i], capture_traces)
        })?;

        let mut scores = Vec::with_capacity(results.len());
        let any_objectives = results.iter().any(|r| r.objectives.is_some());
        let mut objectives = if any_objectives {
            Some(Vec::with_capacity(results.len()))
        } else {
            None
        };
        let mut feedback = if capture_traces {
            Vec::with_capacity(results.len())
        } else {
            Vec::new()
        };
        let mut total_calls = 0usize;
        let mut any_custom_calls = false;

        for r in results {
            scores.push(r.score);
            if let Some(objs) = &mut objectives {
                objs.push(r.objectives.unwrap_or_default());
            }
            if capture_traces {
                feedback.push(r.feedback.unwrap_or(Value::Null));
            }
            if let Some(c) = r.metric_calls {
                total_calls += c;
                any_custom_calls = true;
            } else {
                total_calls += 1;
            }
        }

        Ok(EvalBatch {
            scores,
            objectives,
            feedback,
            metric_calls: if any_custom_calls { Some(total_calls) } else { None },
        })
    }
}

/// Evaluate `items` items with up to `concurrency` worker threads and return results in index order.
pub(crate) fn run_parallel<F>(concurrency: usize, items: usize, f: F) -> gepa_core::Result<Vec<SingleEval>>
where
    F: Fn(usize) -> gepa_core::Result<SingleEval> + Sync,
{
    let workers = concurrency.max(1).min(items.max(1));
    if workers <= 1 || items <= 1 {
        return (0..items).map(f).collect();
    }

    let next = AtomicUsize::new(0);
    let mut slots: Vec<Option<gepa_core::Result<SingleEval>>> = (0..items).map(|_| None).collect();
    let slot_refs: Vec<std::sync::Mutex<Option<gepa_core::Result<SingleEval>>>> =
        slots.drain(..).map(std::sync::Mutex::new).collect();

    thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| loop {
                let idx = next.fetch_add(1, Ordering::Relaxed);
                if idx >= items {
                    break;
                }
                let res = f(idx);
                *slot_refs[idx].lock().unwrap() = Some(res);
            });
        }
    });

    let mut out = Vec::with_capacity(items);
    for slot in slot_refs {
        out.push(slot.into_inner().unwrap().ok_or("missing parallel eval result")??);
    }
    Ok(out)
}

pub(crate) fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::process::Command;
    use std::thread;
    use std::time::Duration;

    use serde_json::json;

    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("gepa_eval_test_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn shell_check_parses_markers_protects_files_and_keeps_parallel_order() {
        let tpl = temp_dir("shell_check");
        fs::write(tpl.join("locked.txt"), b"immutable").unwrap();

        #[cfg(windows)]
        let cmd = "echo GEPA_SCORE=%GEPA_DATA_ID%.0& echo GEPA_SCORES={\"len\":2.0}& type answer.txt".to_string();
        #[cfg(unix)]
        let cmd = "echo GEPA_SCORE=${GEPA_DATA_ID}.0; echo 'GEPA_SCORES={\"len\":2.0}'; cat answer.txt".to_string();

        let eval = AnyEvaluator::new(EvalConfig::ShellCheck(ShellCheckConfig {
            check: cmd,
            targets: BTreeMap::from([("ans".into(), PathBuf::from("answer.txt"))]),
            template_dir: Some(tpl.clone()),
            protect: vec![PathBuf::from("locked.txt")],
            concurrency: 3,
            timeout_secs: 10,
            ..Default::default()
        }));

        let cand = Candidate::from([("ans".into(), "hello".into())]);
        let ids = [3, 1, 2];
        let examples = [json!({"q": 3}), json!({"q": 1}), json!({"q": 2})];
        let batch = eval.evaluate(&cand, Split::Val, &ids, &examples, true).unwrap();
        assert_eq!(batch.scores, vec![3.0, 1.0, 2.0]);
        assert_eq!(batch.feedback.len(), 3);
        assert!(batch.objectives.as_ref().unwrap()[0].get("len") == Some(&2.0));

        // Overwriting a protected file must be rejected with score 0.0.
        let bad_eval = AnyEvaluator::new(EvalConfig::ShellCheck(ShellCheckConfig {
            check: "echo GEPA_SCORE=1.0".into(),
            targets: BTreeMap::from([("ans".into(), PathBuf::from("locked.txt"))]),
            template_dir: Some(tpl.clone()),
            protect: vec![PathBuf::from("locked.txt")],
            ..Default::default()
        }));
        let bad = bad_eval.evaluate(&cand, Split::Train, &[0], &[json!({})], true).unwrap();
        assert_eq!(bad.scores, vec![0.0]);
        assert!(bad.feedback[0]["violation"].as_str().unwrap().contains("locked.txt"));

        let _ = fs::remove_dir_all(&tpl);
    }

    #[test]
    fn http_evaluator_posts_payload_and_parses_response() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let h = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
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
                if let Some(hend) = header_end
                    && buf.len() >= hend + content_len.unwrap_or(0)
                {
                    break;
                }
            }
            let body = r#"{"score":0.75,"scores":{"acc":0.75},"side_info":{"note":"ok"}}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(resp.as_bytes()).unwrap();
            let _ = stream.flush();
        });

        let eval = AnyEvaluator::new(EvalConfig::Http(HttpEvalConfig {
            url: format!("http://{addr}/eval"),
            timeout_secs: 10,
            ..Default::default()
        }));
        let cand = Candidate::from([("p".into(), "hi".into())]);
        let batch = eval.evaluate(&cand, Split::Train, &[0], &[json!({"x": 1})], true).unwrap();
        assert_eq!(batch.scores, vec![0.75]);
        assert_eq!(batch.objectives.unwrap()[0]["acc"], 0.75);
        assert_eq!(batch.feedback[0]["note"], "ok");
        h.join().unwrap();
    }

    #[test]
    fn command_json_parses_last_json_line() {
        #[cfg(windows)]
        let (prog, args) = (
            "powershell".to_string(),
            vec![
                "-NoProfile".into(),
                "-Command".into(),
                "Write-Output 'debug log line'; Write-Output '{\"score\":0.5,\"side_info\":{\"k\":\"v\"}}'".into(),
            ],
        );
        #[cfg(unix)]
        let (prog, args) = (
            "sh".to_string(),
            vec![
                "-c".into(),
                "echo debug log line; echo '{\"score\":0.5,\"side_info\":{\"k\":\"v\"}}'".into(),
            ],
        );

        let eval = AnyEvaluator::new(EvalConfig::CommandJson(JsonCommandConfig {
            program: prog,
            args,
            timeout_secs: 10,
            ..Default::default()
        }));
        let cand = Candidate::from([("p".into(), "hi".into())]);
        let batch = eval.evaluate(&cand, Split::Val, &[0], &[json!({})], true).unwrap();
        assert_eq!(batch.scores, vec![0.5]);
        assert_eq!(batch.feedback[0]["k"], "v");
    }

    #[test]
    fn process_timeout_kills_tree_without_hanging() {
        #[cfg(windows)]
        let mut cmd = {
            let mut c = Command::new("cmd.exe");
            c.args(["/d", "/c", "ping -n 10 127.0.0.1 >nul"]);
            c
        };
        #[cfg(unix)]
        let mut cmd = {
            let mut c = Command::new("sh");
            c.args(["-c", "sleep 10"]);
            c
        };

        let out = run_command_with_timeout(&mut cmd, None, Duration::from_millis(250)).unwrap();
        assert!(out.timed_out);
    }
}
