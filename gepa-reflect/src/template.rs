//! Instruction-proposal prompt rendering and deterministic side-info formatting.

use std::fmt::Write as _;

use gepa_core::{Candidate, DataId, EvalBatch};
use serde::{Deserialize, Serialize};
use serde_json::Value;

fn default_true() -> bool {
    true
}

/// Prompt configuration for reflective mutation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PromptConfig {
    pub objective: Option<String>,
    pub background: Option<String>,
    /// Optional custom template supporting `{component_name}`, `{current_text}`,
    /// `{feedback}`, `{lineage}`, `{objective}`, and `{background}`.
    pub custom_template: Option<String>,
    /// Top-level keys stripped from `EvalBatch.feedback` objects before rendering.
    pub hide_keys: Vec<String>,
    /// Inject full-lineage memory (rejected edits, best branch, sub-metric leaders).
    #[serde(default = "default_true")]
    pub lineage: bool,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            objective: None,
            background: None,
            custom_template: None,
            hide_keys: Vec::new(),
            lineage: true,
        }
    }
}

/// Format minibatch scores, per-example objectives, and `feedback` into Markdown.
///
/// Key order inside `Value::Object` entries follows `serde_json` (`preserve_order`)
/// deterministically while filtering out any keys listed in `hide_keys`.
pub fn format_feedback(ids: &[DataId], eval: &EvalBatch, hide_keys: &[String]) -> String {
    let mut out = String::new();
    for (i, &id) in ids.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let score = eval.scores.get(i).copied().unwrap_or(f64::NAN);
        let _ = writeln!(out, "### Example {} (id={id}, score={score:.4})", i + 1);

        if let Some(objs) = eval.objectives.as_ref().and_then(|all| all.get(i))
            && !objs.is_empty()
        {
            let parts: Vec<String> = objs.iter().map(|(k, v)| format!("{k}={v:.4}")).collect();
            let _ = writeln!(out, "- Objectives: {}", parts.join(", "));
        }

        if let Some(fb) = eval.feedback.get(i) {
            render_feedback_value(&mut out, fb, hide_keys);
        }
    }
    out.trim_end().to_string()
}

fn render_feedback_value(out: &mut String, fb: &Value, hide_keys: &[String]) {
    match fb {
        Value::Null => {}
        Value::Object(map) => {
            for (k, v) in map {
                if hide_keys.iter().any(|h| h == k) || v.is_null() {
                    continue;
                }
                match v {
                    Value::String(s) => {
                        if s.contains('\n') {
                            let _ = writeln!(out, "- {k}:\n{s}");
                        } else {
                            let _ = writeln!(out, "- {k}: {s}");
                        }
                    }
                    other => {
                        let rendered = serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string());
                        if rendered.contains('\n') {
                            let _ = writeln!(out, "- {k}:\n{rendered}");
                        } else {
                            let _ = writeln!(out, "- {k}: {rendered}");
                        }
                    }
                }
            }
        }
        Value::String(s) => {
            let _ = writeln!(out, "{s}");
        }
        other => {
            let rendered = serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string());
            let _ = writeln!(out, "{rendered}");
        }
    }
}

/// Choose a fence at least 3 backticks long (4+ if `text` already contains triple backticks).
fn fence_for(text: &str) -> String {
    let mut max_run = 0usize;
    let mut cur = 0usize;
    for b in text.bytes() {
        if b == b'`' {
            cur += 1;
            max_run = max_run.max(cur);
        } else {
            cur = 0;
        }
    }
    "`".repeat(max_run.max(2) + 1)
}

/// Render the reflection prompt for updating `component` of `parent`.
pub fn render_prompt(
    config: &PromptConfig,
    parent: &Candidate,
    component: &str,
    ids: &[DataId],
    eval: &EvalBatch,
) -> String {
    render_prompt_with_lineage(config, parent, component, ids, eval, "")
}

/// Render the reflection prompt with optional full-lineage context injected before `## Task`.
pub fn render_prompt_with_lineage(
    config: &PromptConfig,
    parent: &Candidate,
    component: &str,
    ids: &[DataId],
    eval: &EvalBatch,
    lineage_ctx: &str,
) -> String {
    let current_text = parent.get(component).map(String::as_str).unwrap_or("");
    let feedback = format_feedback(ids, eval, &config.hide_keys);
    let objective = config.objective.as_deref().unwrap_or("");
    let background = config.background.as_deref().unwrap_or("");

    if let Some(tpl) = &config.custom_template {
        return tpl
            .replace("{component_name}", component)
            .replace("{current_text}", current_text)
            .replace("{feedback}", &feedback)
            .replace("{lineage}", lineage_ctx)
            .replace("{objective}", objective)
            .replace("{background}", background);
    }

    let fence = fence_for(current_text);
    let mut out = String::new();
    let _ = writeln!(
        out,
        "You are optimizing the text component `{component}` of a candidate artifact."
    );
    if !objective.is_empty() {
        let _ = writeln!(out, "\n## Objective\n{objective}");
    }
    if !background.is_empty() {
        let _ = writeln!(out, "\n## Background\n{background}");
    }

    if parent.len() > 1 {
        let _ = writeln!(out, "\n## Other Components (read-only context)");
        for (name, val) in parent {
            if name == component {
                continue;
            }
            let other_fence = fence_for(val);
            let _ = writeln!(out, "### `{name}`\n{other_fence}\n{val}\n{other_fence}");
        }
    }

    let _ = writeln!(
        out,
        "\n## Current `{component}`\n{fence}\n{current_text}\n{fence}"
    );
    let _ = writeln!(out, "\n## Evaluation Feedback\n{feedback}");
    if !lineage_ctx.is_empty() {
        let _ = writeln!(out, "\n## Lineage Memory\n{lineage_ctx}");
    }
    let _ = write!(
        out,
        "\n## Task\nDiagnose why the current `{component}` failed or lost points in the feedback above, and write an improved replacement for `{component}`.\nReturn ONLY the complete updated text of `{component}` inside a single {fence} ... {fence} fenced code block."
    );
    out
}
