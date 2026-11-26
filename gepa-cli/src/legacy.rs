//! Backward-compatibility layer and migration converter for legacy `gepa_plugin` (`tasks/*.json`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use gepa_core::strategies::Stop;
use gepa_eval::{EvalConfig, JsonCommandConfig, ShellCheckConfig};
use gepa_reflect::{LmConfig, PromptConfig};
use serde::Deserialize;
use serde_json::Value;

use crate::schema::{DataMode, DatasetSpec, ReflectorSpec, TaskSpec, default_engine_config};

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyTask {
    pub name: Option<String>,
    pub mode: Option<String>,
    pub candidate: Option<String>,
    pub target: Option<String>,
    pub data: Option<PathBuf>,
    pub val: Option<PathBuf>,
    pub test: Option<Value>,
    pub score: Option<String>,
    pub check: Option<String>,
    pub workspace: Option<PathBuf>,
    pub protect: Vec<PathBuf>,
    pub objective: Option<String>,
    pub background: Option<String>,
    pub seed: Option<Value>,
    pub budget: Option<usize>,
    pub stop_at_score: Option<f64>,
    pub check_timeout: Option<u64>,
}

/// Returns `true` when `raw` appears to follow the legacy `gepa_plugin/tasks/*.json` schema.
pub fn is_legacy_json(raw: &Value) -> bool {
    let Some(obj) = raw.as_object() else {
        return false;
    };
    !obj.contains_key("evaluator")
        && (obj.contains_key("score")
            || obj.contains_key("check")
            || obj.contains_key("candidate"))
}

/// Convert a legacy `tasks/*.json` object into the universal [`TaskSpec`] plus actionable warnings.
pub fn convert_legacy(
    legacy: LegacyTask,
    task_dir: &Path,
    python_bin: Option<&str>,
) -> gepa_core::Result<(TaskSpec, Vec<String>)> {
    let mut warnings = Vec::new();
    let mode = match legacy.mode.as_deref().unwrap_or("generalize") {
        "generalize" => DataMode::Generalize,
        "multi" => DataMode::Multi,
        "single" => DataMode::Single,
        other => return Err(format!("unknown legacy mode `{other}`").into()),
    };

    let candidate_kind = legacy.candidate.as_deref().unwrap_or(match mode {
        DataMode::Generalize => "agents_md",
        DataMode::Multi | DataMode::Single => "file",
    });

    let default_comp_key = if candidate_kind == "agents_md" {
        "agents_md".to_string()
    } else {
        "target".to_string()
    };

    let (seed, seed_files) =
        normalize_seed(legacy.seed, &default_comp_key, candidate_kind, legacy.workspace.as_deref(), legacy.target.as_deref())?;
    let comp_keys: Vec<String> = seed.keys().chain(seed_files.keys()).cloned().collect();
    let python = python_bin.unwrap_or("python");
    let score_kind = legacy.score.as_deref().unwrap_or("command");

    let evaluator = if candidate_kind == "file" && (score_kind == "command" || legacy.check.is_some()) {
        let raw_check = legacy.check.ok_or("legacy command/file task is missing `check`")?;
        let check = raw_check.replace("{python}", python);
        if raw_check.contains("{python}") && python_bin.is_none() {
            warnings.push(
                "Replaced `{python}` placeholder with `python`; adjust `evaluator.check` or pass `--python` if a virtualenv interpreter is required.".into(),
            );
        }

        let target_rel = PathBuf::from(legacy.target.as_deref().unwrap_or("CANDIDATE.txt"));
        let mut targets = BTreeMap::new();
        for key in &comp_keys {
            if comp_keys.len() == 1 {
                targets.insert(key.clone(), target_rel.clone());
            } else {
                targets.insert(key.clone(), PathBuf::from(key));
            }
        }

        let mut sc = ShellCheckConfig {
            check,
            targets,
            template_dir: legacy.workspace,
            protect: legacy.protect,
            task_dir: Some(task_dir.to_path_buf()),
            ..Default::default()
        };
        if let Some(ct) = legacy.check_timeout {
            sc.timeout_secs = ct;
        }
        EvalConfig::ShellCheck(sc)
    } else {
        warnings.push(format!(
            "Legacy `{score_kind}` / `{candidate_kind}` task requires an external agent evaluator; mapped to a placeholder `command_json` evaluator (`{python} eval_wrapper.py`). Point `evaluator.program` / `evaluator.args` at your agent evaluator."
        ));
        EvalConfig::CommandJson(JsonCommandConfig {
            program: python.to_string(),
            args: vec!["eval_wrapper.py".to_string()],
            ..Default::default()
        })
    };

    let mut engine = default_engine_config();
    engine.stop = Stop {
        max_metric_calls: Some(legacy.budget.unwrap_or(150)),
        score_threshold: legacy.stop_at_score,
        ..Default::default()
    };
    if mode == DataMode::Single {
        engine.minibatch_size = 1;
    }

    let mut hide_keys = Vec::new();
    if matches!(mode, DataMode::Multi | DataMode::Single) {
        hide_keys = vec![
            "Expected answer".to_string(),
            "expected".to_string(),
            "answer".to_string(),
        ];
    }

    let test = parse_legacy_test(legacy.test, &mut warnings)?;
    let dataset = if mode == DataMode::Generalize && legacy.val.is_none() {
        DatasetSpec {
            pool: legacy.data,
            test,
            ..Default::default()
        }
    } else {
        DatasetSpec {
            train: legacy.data,
            val: legacy.val,
            test,
            ..Default::default()
        }
    };

    let spec = TaskSpec {
        schema_version: 1,
        name: legacy.name,
        mode,
        seed,
        seed_files,
        dataset,
        evaluator,
        reflector: ReflectorSpec {
            lm: LmConfig::default(),
            prompt: PromptConfig {
                objective: legacy.objective,
                background: legacy.background,
                hide_keys,
                ..Default::default()
            },
        },
        engine,
    };

    Ok((spec, warnings))
}

fn parse_legacy_test(raw: Option<Value>, warnings: &mut Vec<String>) -> gepa_core::Result<Option<PathBuf>> {
    match raw {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(PathBuf::from(s))),
        Some(Value::Array(items)) => {
            if items.is_empty() {
                return Ok(None);
            }
            if items.len() > 1 {
                warnings.push(format!(
                    "Legacy `test` array has {} files; only the first (`{}`) was mapped to `dataset.test`.",
                    items.len(),
                    items[0]
                ));
            }
            let first = items[0]
                .as_str()
                .ok_or("legacy `test` array elements must be string paths")?;
            Ok(Some(PathBuf::from(first)))
        }
        Some(_) => Err("legacy `test` must be a string path or array of string paths".into()),
    }
}

fn normalize_seed(
    seed: Option<Value>,
    default_key: &str,
    candidate_kind: &str,
    workspace: Option<&Path>,
    target: Option<&str>,
) -> gepa_core::Result<(BTreeMap<String, String>, BTreeMap<String, PathBuf>)> {
    match seed {
        None | Some(Value::Null) => {
            if candidate_kind == "file"
                && let (Some(ws), Some(tgt)) = (workspace, target)
            {
                return Ok((
                    BTreeMap::new(),
                    BTreeMap::from([(default_key.to_string(), ws.join(tgt))]),
                ));
            }
            Ok((
                BTreeMap::from([(default_key.to_string(), "You are a helpful assistant.".to_string())]),
                BTreeMap::new(),
            ))
        }
        Some(Value::String(s)) => Ok((BTreeMap::from([(default_key.to_string(), s)]), BTreeMap::new())),
        Some(Value::Object(map)) => {
            let mut out = BTreeMap::new();
            for (k, v) in map {
                let text = v
                    .as_str()
                    .ok_or_else(|| format!("legacy seed component `{k}` must be a string"))?;
                out.insert(k, text.to_string());
            }
            Ok((out, BTreeMap::new()))
        }
        Some(_) => Err("legacy `seed` must be a string or object map".into()),
    }
}
