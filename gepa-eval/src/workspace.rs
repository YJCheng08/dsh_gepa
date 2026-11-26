//! Isolated workspace + shell check evaluator (`shell_check`).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use gepa_core::state::Objectives;
use gepa_core::{Candidate, DataId, Split};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::process::run_command_with_timeout;
use crate::{SingleEval, truncate_utf8};

static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

fn default_timeout_secs() -> u64 {
    300
}

fn default_concurrency() -> usize {
    1
}

/// Configuration for writing candidate files into an isolated workspace and running a check command.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ShellCheckConfig {
    /// Shell command to execute inside the workspace. Supports `{task_dir}` and `{workspace}`.
    pub check: String,
    /// Mapping of candidate component name -> relative file path inside the workspace.
    pub targets: BTreeMap<String, PathBuf>,
    /// Optional template workspace directory copied into a fresh temporary directory per evaluation.
    pub template_dir: Option<PathBuf>,
    /// Relative paths inside the workspace that must not be overwritten or modified.
    pub protect: Vec<PathBuf>,
    /// Base task directory used for `{task_dir}` placeholder expansion.
    pub task_dir: Option<PathBuf>,
    pub env: BTreeMap<String, String>,
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
}

impl Default for ShellCheckConfig {
    fn default() -> Self {
        Self {
            check: String::new(),
            targets: BTreeMap::new(),
            template_dir: None,
            protect: Vec::new(),
            task_dir: None,
            env: BTreeMap::new(),
            timeout_secs: default_timeout_secs(),
            concurrency: default_concurrency(),
        }
    }
}

struct TempWorkspace {
    path: PathBuf,
}

impl TempWorkspace {
    fn create() -> gepa_core::Result<Self> {
        let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("gepa_ws_{pid}_{nanos}_{seq}"));
        fs::create_dir_all(&path)?;
        Ok(Self { path })
    }
}

impl Drop for TempWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn validate_rel_path(rel: &Path) -> gepa_core::Result<PathBuf> {
    if rel.as_os_str().is_empty() || rel.is_absolute() {
        return Err(format!("workspace relative path required, got {}", rel.display()).into());
    }
    let mut clean = PathBuf::new();
    for comp in rel.components() {
        match comp {
            Component::Normal(p) => clean.push(p),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(format!("path traversal is not allowed in workspace path {}", rel.display()).into());
            }
        }
    }
    if clean.as_os_str().is_empty() {
        return Err(format!("empty normalized relative path {}", rel.display()).into());
    }
    Ok(clean)
}

fn copy_dir_filtered(src: &Path, dst: &Path) -> gepa_core::Result<()> {
    if !src.exists() {
        return Err(format!("template_dir does not exist: {}", src.display()).into());
    }
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == ".git" || name == "target" {
            continue;
        }
        let src_path = entry.path();
        let dst_path = dst.join(&name);
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_dir_filtered(&src_path, &dst_path)?;
        } else if ty.is_file() {
            fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

fn resolve_targets(cfg: &ShellCheckConfig, candidate: &Candidate) -> gepa_core::Result<Vec<(PathBuf, String)>> {
    if cfg.targets.is_empty() {
        if candidate.len() == 1 {
            let (_, text) = candidate.iter().next().unwrap();
            return Ok(vec![(PathBuf::from("CANDIDATE.txt"), text.clone())]);
        }
        return Err("shell_check evaluator requires `targets` mapping when candidate has multiple components".into());
    }

    let mut resolved = Vec::with_capacity(cfg.targets.len());
    for (comp, rel) in &cfg.targets {
        let text = candidate
            .get(comp)
            .ok_or_else(|| format!("target component `{comp}` missing from candidate"))?;
        resolved.push((validate_rel_path(rel)?, text.clone()));
    }
    Ok(resolved)
}

pub(crate) fn eval_one(
    cfg: &ShellCheckConfig,
    candidate: &Candidate,
    split: Split,
    id: DataId,
    example: &Value,
    capture_traces: bool,
) -> gepa_core::Result<SingleEval> {
    let raw_check = example
        .get("check")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(&cfg.check);
    if raw_check.trim().is_empty() {
        return Err("shell_check evaluator requires a non-empty `check` command".into());
    }

    let ws = TempWorkspace::create()?;
    if let Some(tpl) = &cfg.template_dir {
        copy_dir_filtered(tpl, &ws.path)?;
    }

    let mut protected = Vec::with_capacity(cfg.protect.len());
    for p in &cfg.protect {
        let clean = validate_rel_path(p)?;
        let full = ws.path.join(&clean);
        let snap = if full.exists() { Some(fs::read(&full)?) } else { None };
        protected.push((clean, snap));
    }

    let targets = resolve_targets(cfg, candidate)?;
    for (clean_rel, text) in &targets {
        if let Some((violated, _)) = protected.iter().find(|(p, _)| p == clean_rel) {
            return Ok(protection_violation(
                &format!("target attempted to overwrite protected path `{}`", violated.display()),
                example,
                capture_traces,
            ));
        }
        let dest = ws.path.join(clean_rel);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&dest, text.as_bytes())?;
    }

    let task_dir_str = cfg
        .task_dir
        .as_deref()
        .unwrap_or_else(|| Path::new("."))
        .to_string_lossy();
    let ws_str = ws.path.to_string_lossy();
    let expanded_check = raw_check
        .replace("{python}", "python")
        .replace("{task_dir}", &task_dir_str)
        .replace("{workspace}", &ws_str);

    let example_json = serde_json::to_string(example)?;
    let split_str = match split {
        Split::Train => "train",
        Split::Val => "val",
    };

    let mut cmd = shell_command(&expanded_check);
    cmd.current_dir(&ws.path)
        .env("GEPA_DATA_ID", id.to_string())
        .env("GEPA_SPLIT", split_str)
        .env("GEPA_EXAMPLE_JSON", &example_json);
    for (k, v) in &cfg.env {
        cmd.env(k, v);
    }

    let out = run_command_with_timeout(
        &mut cmd,
        Some(example_json.as_bytes()),
        Duration::from_secs(cfg.timeout_secs.max(1)),
    )?;

    for (clean_rel, expected) in &protected {
        let full = ws.path.join(clean_rel);
        let actual = if full.exists() { Some(fs::read(&full)?) } else { None };
        if actual.as_ref() != expected.as_ref() {
            return Ok(protection_violation(
                &format!("protected path `{}` was modified or deleted during check", clean_rel.display()),
                example,
                capture_traces,
            ));
        }
    }

    let (parsed_score, objectives) = parse_gepa_markers(&out.stdout);
    let score = if out.timed_out {
        0.0
    } else if let Some(s) = parsed_score {
        s
    } else if out.exit_code == Some(0) {
        1.0
    } else {
        0.0
    };

    let feedback = if capture_traces {
        let mut m = Map::new();
        if example.as_object().is_some_and(|o| !o.is_empty()) {
            m.insert("example".into(), example.clone());
        }
        m.insert("score".into(), json!(score));
        m.insert("exit_code".into(), json!(out.exit_code));
        if out.timed_out {
            m.insert("timed_out".into(), json!(true));
        }
        if !out.stdout.is_empty() {
            m.insert("stdout".into(), Value::String(truncate_utf8(&out.stdout, 16 * 1024).into()));
        }
        if !out.stderr.is_empty() {
            m.insert("stderr".into(), Value::String(truncate_utf8(&out.stderr, 16 * 1024).into()));
        }
        Some(Value::Object(m))
    } else {
        None
    };

    Ok(SingleEval {
        score,
        objectives,
        feedback,
        metric_calls: Some(1),
    })
}

fn protection_violation(msg: &str, example: &Value, capture_traces: bool) -> SingleEval {
    let feedback = if capture_traces {
        let mut m = Map::new();
        if example.as_object().is_some_and(|o| !o.is_empty()) {
            m.insert("example".into(), example.clone());
        }
        m.insert("score".into(), json!(0.0));
        m.insert("violation".into(), Value::String(msg.to_string()));
        Some(Value::Object(m))
    } else {
        None
    };
    SingleEval {
        score: 0.0,
        objectives: None,
        feedback,
        metric_calls: Some(1),
    }
}

fn shell_command(script: &str) -> Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let mut c = Command::new("cmd.exe");
        // Use raw_arg so inner double quotes are not backslash-escaped by MSVC argv quoting.
        c.raw_arg(format!("/d /s /c \"{script}\""));
        c
    }
    #[cfg(unix)]
    {
        let mut c = Command::new("sh");
        c.args(["-c", script]);
        c
    }
}

/// Parse `GEPA_SCORE=<f64>` and `GEPA_SCORES={<json>}` lines from stdout (last match wins).
pub(crate) fn parse_gepa_markers(stdout: &str) -> (Option<f64>, Option<Objectives>) {
    let mut score = None;
    let mut objectives = None;
    for raw_line in stdout.lines() {
        let line = raw_line.trim();
        if let Some(rest) = line.strip_prefix("GEPA_SCORE=") {
            if let Ok(val) = rest.trim().parse::<f64>()
                && val.is_finite()
            {
                score = Some(val);
            }
        } else if let Some(rest) = line.strip_prefix("GEPA_SCORES=")
            && let Ok(map) = serde_json::from_str::<BTreeMap<String, f64>>(rest.trim())
            && map.values().all(|v| v.is_finite())
        {
            objectives = Some(map);
        }
    }
    (score, objectives)
}
