//! Universal cross-platform task schema and loader (`task.json`).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use clap::ValueEnum;
use gepa_core::Candidate;
use gepa_core::rng::Rng;
use gepa_core::strategies::Stop;
use gepa_eval::EvalConfig;
use gepa_reflect::{LmConfig, PromptConfig};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::legacy;

fn default_schema_version() -> u32 {
    1
}

pub fn default_engine_config() -> gepa_core::Config {
    gepa_core::Config {
        stop: Stop {
            max_metric_calls: Some(150),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum DataMode {
    #[default]
    Generalize,
    Multi,
    Single,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DatasetSpec {
    pub train: Option<PathBuf>,
    pub val: Option<PathBuf>,
    pub test: Option<PathBuf>,
    /// Single JSONL/JSON pool deterministically split into train/val(/test) in `generalize` mode.
    pub pool: Option<PathBuf>,
    pub inline_train: Option<Vec<Value>>,
    pub inline_val: Option<Vec<Value>>,
    pub inline_test: Option<Vec<Value>>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ReflectorSpec {
    pub lm: LmConfig,
    #[serde(flatten)]
    pub prompt: PromptConfig,
}

/// Universal GEPA task configuration (`schema_version = 1`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TaskSpec {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub name: Option<String>,
    pub mode: DataMode,
    pub seed: BTreeMap<String, String>,
    pub seed_files: BTreeMap<String, PathBuf>,
    pub dataset: DatasetSpec,
    pub evaluator: EvalConfig,
    pub reflector: ReflectorSpec,
    #[serde(default = "default_engine_config")]
    pub engine: gepa_core::Config,
}

impl Default for TaskSpec {
    fn default() -> Self {
        Self {
            schema_version: default_schema_version(),
            name: None,
            mode: DataMode::Generalize,
            seed: BTreeMap::new(),
            seed_files: BTreeMap::new(),
            dataset: DatasetSpec::default(),
            evaluator: EvalConfig::ShellCheck(Default::default()),
            reflector: ReflectorSpec::default(),
            engine: default_engine_config(),
        }
    }
}

/// Resolved task ready for execution.
#[derive(Clone, Debug)]
pub struct LoadedTask {
    pub spec: TaskSpec,
    pub seed: Candidate,
    pub train_data: Vec<Value>,
    pub val_data: Vec<Value>,
    pub test_data: Vec<Value>,
    #[allow(dead_code)]
    pub task_dir: PathBuf,
    pub warnings: Vec<String>,
}

pub fn load_task_from_file(path: &Path) -> gepa_core::Result<LoadedTask> {
    let raw_text = fs::read_to_string(path)
        .map_err(|e| format!("failed to read task file `{}`: {e}", path.display()))?;
    let raw_val: Value = serde_json::from_str(&raw_text)
        .map_err(|e| format!("invalid JSON in `{}`: {e}", path.display()))?;

    let task_dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let (mut spec, warnings) = if legacy::is_legacy_json(&raw_val) {
        let leg: legacy::LegacyTask = serde_json::from_value(raw_val)?;
        legacy::convert_legacy(leg, &task_dir, None)?
    } else {
        let spec: TaskSpec = serde_json::from_value(raw_val)?;
        (spec, Vec::new())
    };

    resolve_loaded_task(&mut spec, &task_dir, warnings)
}

pub fn resolve_loaded_task(
    spec: &mut TaskSpec,
    task_dir: &Path,
    warnings: Vec<String>,
) -> gepa_core::Result<LoadedTask> {
    let mut seed = spec.seed.clone();
    for (comp, rel) in &spec.seed_files {
        if seed.contains_key(comp) {
            return Err(format!("component `{comp}` defined in both `seed` and `seed_files`").into());
        }
        let resolved = resolve_path(task_dir, rel);
        let content = fs::read_to_string(&resolved)
            .map_err(|e| format!("failed reading seed file `{}` for `{comp}`: {e}", resolved.display()))?;
        seed.insert(comp.clone(), content);
    }
    if seed.is_empty() {
        return Err("task `seed` (or `seed_files`) must contain at least one component".into());
    }

    resolve_evaluator_paths(&mut spec.evaluator, task_dir);
    if let Some(stop_file) = spec.engine.stop.stop_file.as_mut() {
        *stop_file = resolve_path(task_dir, stop_file);
    }

    let (train_data, val_data, test_data) = load_datasets(spec, task_dir)?;
    if spec.mode == DataMode::Multi && spec.reflector.prompt.hide_keys.is_empty() {
        spec.reflector.prompt.hide_keys = vec![
            "expected".to_string(),
            "Expected answer".to_string(),
            "answer".to_string(),
        ];
    }
    if spec.mode == DataMode::Single {
        spec.engine.minibatch_size = 1;
    }

    Ok(LoadedTask {
        spec: spec.clone(),
        seed,
        train_data,
        val_data,
        test_data,
        task_dir: task_dir.to_path_buf(),
        warnings,
    })
}

fn load_datasets(spec: &TaskSpec, task_dir: &Path) -> gepa_core::Result<(Vec<Value>, Vec<Value>, Vec<Value>)> {
    let load_opt = |inline: &Option<Vec<Value>>, path: &Option<PathBuf>| -> gepa_core::Result<Option<Vec<Value>>> {
        if let Some(items) = inline {
            return Ok(Some(items.clone()));
        }
        if let Some(p) = path {
            let full = resolve_path(task_dir, p);
            return Ok(Some(read_dataset_file(&full)?));
        }
        Ok(None)
    };

    let train = load_opt(&spec.dataset.inline_train, &spec.dataset.train)?;
    let val = load_opt(&spec.dataset.inline_val, &spec.dataset.val)?;
    let test = load_opt(&spec.dataset.inline_test, &spec.dataset.test)?.unwrap_or_default();

    match spec.mode {
        DataMode::Generalize => {
            if train.is_none() && val.is_none()
                && let Some(pool_path) = &spec.dataset.pool
            {
                let mut pool = read_dataset_file(&resolve_path(task_dir, pool_path))?;
                if pool.len() < 2 {
                    return Err("generalize mode `dataset.pool` must contain at least 2 examples".into());
                }
                Rng::new(0).shuffle(&mut pool);
                if !test.is_empty() {
                    let mid = pool.len() / 2;
                    let val_data = pool.split_off(mid);
                    return Ok((pool, val_data, test));
                }
                let n = pool.len();
                let t_end = (n / 3).max(1);
                let v_end = ((2 * n) / 3).max(t_end + 1).min(n);
                let test_split = pool.split_off(v_end);
                let val_data = pool.split_off(t_end);
                return Ok((pool, val_data, test_split));
            }
            let train_data = train.ok_or(
                "generalize mode requires `dataset.train` (or `dataset.pool` / `dataset.inline_train`)",
            )?;
            let val_data = val.ok_or(
                "generalize mode requires `dataset.val` (or `dataset.pool` / `dataset.inline_val`); set `mode: \"multi\"` if using a single shared dataset",
            )?;
            if train_data.is_empty() || val_data.is_empty() {
                return Err("generalize mode datasets (`train` and `val`) must be non-empty".into());
            }
            Ok((train_data, val_data, test))
        }
        DataMode::Multi => {
            let shared = train
                .or(val)
                .ok_or("multi mode requires `dataset.train` or `dataset.val`")?;
            if shared.is_empty() {
                return Err("multi mode dataset must be non-empty".into());
            }
            Ok((shared.clone(), shared, test))
        }
        DataMode::Single => {
            let single = vec![json!({})];
            Ok((single.clone(), single, test))
        }
    }
}

fn read_dataset_file(path: &Path) -> gepa_core::Result<Vec<Value>> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("failed to read dataset `{}`: {e}", path.display()))?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    if trimmed.starts_with('[') {
        let items: Vec<Value> = serde_json::from_str(trimmed)
            .map_err(|e| format!("invalid JSON array dataset `{}`: {e}", path.display()))?;
        return Ok(items);
    }
    let mut items = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let val: Value = serde_json::from_str(t)
            .map_err(|e| format!("invalid JSONL in `{}` line {}: {e}", path.display(), idx + 1))?;
        items.push(val);
    }
    Ok(items)
}

fn resolve_path(base: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    }
}

fn resolve_evaluator_paths(eval: &mut EvalConfig, task_dir: &Path) {
    match eval {
        EvalConfig::CommandJson(c) => {
            if let Some(cwd) = c.cwd.as_mut() {
                *cwd = resolve_path(task_dir, cwd);
            }
        }
        EvalConfig::ShellCheck(s) => {
            if let Some(tpl) = s.template_dir.as_mut() {
                *tpl = resolve_path(task_dir, tpl);
            }
            match s.task_dir.as_mut() {
                Some(td) => *td = resolve_path(task_dir, td),
                None => s.task_dir = Some(task_dir.to_path_buf()),
            }
        }
        EvalConfig::Http(_) => {}
    }
}
