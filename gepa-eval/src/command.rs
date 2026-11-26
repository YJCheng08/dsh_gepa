//! Subprocess JSON-protocol evaluator (`command_json`).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use gepa_core::{Candidate, DataId, Split};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::process::run_command_with_timeout;
use crate::{SingleEval, truncate_utf8};

fn default_timeout_secs() -> u64 {
    300
}

fn default_concurrency() -> usize {
    1
}

fn default_fail_score() -> Option<f64> {
    Some(0.0)
}

/// Configuration for an external command speaking JSON over stdin/stdout.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct JsonCommandConfig {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub env: BTreeMap<String, String>,
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    /// Score assigned when the command times out or fails without emitting valid JSON.
    /// Set to `null` to propagate failures as evaluator errors instead.
    #[serde(default = "default_fail_score")]
    pub fail_score: Option<f64>,
}

impl Default for JsonCommandConfig {
    fn default() -> Self {
        Self {
            program: String::new(),
            args: Vec::new(),
            cwd: None,
            env: BTreeMap::new(),
            timeout_secs: default_timeout_secs(),
            concurrency: default_concurrency(),
            fail_score: default_fail_score(),
        }
    }
}

pub(crate) fn eval_one(
    cfg: &JsonCommandConfig,
    candidate: &Candidate,
    split: Split,
    id: DataId,
    example: &Value,
    capture_traces: bool,
) -> gepa_core::Result<SingleEval> {
    if cfg.program.trim().is_empty() {
        return Err("command_json evaluator requires a non-empty `program`".into());
    }

    let payload = json!({
        "candidate": candidate,
        "split": split,
        "id": id,
        "example": example,
        "capture_traces": capture_traces,
    });
    let stdin_bytes = serde_json::to_vec(&payload)?;

    let mut cmd = Command::new(&cfg.program);
    cmd.args(&cfg.args);
    if let Some(cwd) = &cfg.cwd {
        cmd.current_dir(cwd);
    }
    for (k, v) in &cfg.env {
        cmd.env(k, v);
    }

    let out = run_command_with_timeout(&mut cmd, Some(&stdin_bytes), Duration::from_secs(cfg.timeout_secs.max(1)))?;
    if !out.timed_out
        && let Ok(parsed) = parse_json_eval_response(&out.stdout, capture_traces)
    {
        return Ok(parsed);
    }

    if let Some(score) = cfg.fail_score {
        let feedback = if capture_traces {
            let mut m = Map::new();
            m.insert(
                "error".into(),
                Value::String(if out.timed_out {
                    format!("evaluator timed out after {}s", cfg.timeout_secs)
                } else {
                    format!("evaluator exited with {:?} without valid JSON", out.exit_code)
                }),
            );
            if !out.stdout.trim().is_empty() {
                m.insert("stdout".into(), Value::String(truncate_utf8(&out.stdout, 16 * 1024).into()));
            }
            if !out.stderr.trim().is_empty() {
                m.insert("stderr".into(), Value::String(truncate_utf8(&out.stderr, 16 * 1024).into()));
            }
            Some(Value::Object(m))
        } else {
            None
        };
        return Ok(SingleEval {
            score,
            objectives: None,
            feedback,
            metric_calls: Some(1),
        });
    }

    Err(format!(
        "command_json `{}` failed (timed_out={}, exit={:?}): {}",
        cfg.program,
        out.timed_out,
        out.exit_code,
        truncate_utf8(if out.stderr.is_empty() { &out.stdout } else { &out.stderr }, 512)
    )
    .into())
}

/// Parse `{score, objectives|scores, side_info|feedback, metric_calls}` from full stdout
/// or from the last JSON-object line in stdout.
pub(crate) fn parse_json_eval_response(stdout: &str, capture_traces: bool) -> Result<SingleEval, String> {
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Err("empty evaluator output".into());
    }

    let val: Value = match serde_json::from_str::<Value>(trimmed) {
        Ok(v) if v.is_object() => v,
        _ => trimmed
            .lines()
            .rev()
            .map(str::trim)
            .filter(|l| l.starts_with('{'))
            .find_map(|l| serde_json::from_str::<Value>(l).ok().filter(|v| v.is_object()))
            .ok_or_else(|| "no JSON object found in evaluator output".to_string())?,
    };

    let obj = val.as_object().ok_or("evaluator JSON must be an object")?;
    let score = obj
        .get("score")
        .and_then(|v| v.as_f64())
        .filter(|s| s.is_finite())
        .ok_or_else(|| "missing or non-finite numeric `score` field".to_string())?;

    let raw_objs = obj.get("objectives").or_else(|| obj.get("scores"));
    let objectives = match raw_objs {
        Some(Value::Object(map)) => {
            let mut parsed = BTreeMap::new();
            for (k, v) in map {
                let num = v
                    .as_f64()
                    .filter(|s| s.is_finite())
                    .ok_or_else(|| format!("non-numeric objective `{k}`"))?;
                parsed.insert(k.clone(), num);
            }
            Some(parsed)
        }
        Some(Value::Null) | None => None,
        Some(_) => return Err("`objectives` / `scores` must be a JSON object".into()),
    };

    let feedback = if capture_traces {
        let raw_fb = obj.get("side_info").or_else(|| obj.get("feedback"));
        match raw_fb {
            Some(v) if !v.is_null() => Some(v.clone()),
            _ => {
                let mut m = Map::new();
                m.insert("score".into(), json!(score));
                m.insert("stdout".into(), Value::String(truncate_utf8(stdout, 16 * 1024).into()));
                Some(Value::Object(m))
            }
        }
    } else {
        None
    };

    let metric_calls = obj
        .get("metric_calls")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);

    Ok(SingleEval {
        score,
        objectives,
        feedback,
        metric_calls,
    })
}
