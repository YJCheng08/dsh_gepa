//! Full-lineage evaluation memory: records all tried candidates (accepted and rejected)
//! and builds compact failure, cross-branch, and sub-metric context for reflection.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use gepa_core::{Candidate, EvalBatch, Split, State};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalRecord {
    pub texts: Candidate,
    pub parent: Option<usize>,
    pub score: f64,
    pub objectives: BTreeMap<String, f64>,
    pub status: String,
    pub accepted: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Lineage {
    pub records: Vec<EvalRecord>,
    /// Indices into `records` in evaluation order (oldest -> newest).
    pub recent: Vec<usize>,
    #[serde(skip)]
    pending_parents: Vec<(Candidate, Candidate)>,
}

impl Lineage {
    pub fn note_proposal(&mut self, parent: &Candidate, child: &Candidate) {
        self.pending_parents.retain(|(c, _)| c != child);
        self.pending_parents.push((child.clone(), parent.clone()));
        if self.pending_parents.len() > 32 {
            self.pending_parents.remove(0);
        }
    }

    pub fn find(&self, candidate: &Candidate) -> Option<&EvalRecord> {
        self.records.iter().find(|r| r.texts == *candidate)
    }

    fn find_idx(&self, candidate: &Candidate) -> Option<usize> {
        self.records.iter().position(|r| r.texts == *candidate)
    }

    pub fn record_eval(&mut self, candidate: &Candidate, split: Split, eval: &EvalBatch) {
        let score = mean_score(&eval.scores);
        let objectives = mean_objectives(eval.objectives.as_deref());
        let status = summarize_status(&eval.feedback);

        let parent_idx = self
            .pending_parents
            .iter()
            .rev()
            .find(|(c, _)| c == candidate)
            .and_then(|(_, p)| self.find_idx(p));

        if let Some(idx) = self.find_idx(candidate) {
            let rec = &mut self.records[idx];
            if split == Split::Val || !rec.accepted {
                rec.score = score;
                if !objectives.is_empty() {
                    rec.objectives = objectives;
                }
            }
            if status != "ok" || rec.status.is_empty() {
                rec.status = status;
            }
            if rec.parent.is_none() {
                rec.parent = parent_idx;
            }
            self.recent.retain(|&i| i != idx);
            self.recent.push(idx);
        } else {
            let idx = self.records.len();
            self.records.push(EvalRecord {
                texts: candidate.clone(),
                parent: parent_idx,
                score,
                objectives,
                status,
                accepted: false,
            });
            self.recent.push(idx);
        }
    }

    /// Synchronize accepted candidates, validation scores, and parent links from `State`.
    pub fn sync_state(&mut self, state: &State) {
        let mut idx_map = Vec::with_capacity(state.candidates.len());
        for (i, cand) in state.candidates.iter().enumerate() {
            let score = state.val_average(i);
            let objectives = state.objective_scores.get(i).cloned().unwrap_or_default();
            if let Some(rec_idx) = self.find_idx(cand) {
                let rec = &mut self.records[rec_idx];
                rec.accepted = true;
                rec.score = score;
                if !objectives.is_empty() {
                    rec.objectives = objectives;
                }
                idx_map.push(rec_idx);
            } else {
                let rec_idx = self.records.len();
                self.records.push(EvalRecord {
                    texts: cand.clone(),
                    parent: None,
                    score,
                    objectives,
                    status: "ok".to_string(),
                    accepted: true,
                });
                self.recent.push(rec_idx);
                idx_map.push(rec_idx);
            }
        }
        for (i, parents) in state.parents.iter().enumerate() {
            if let Some(&p) = parents.first()
                && let (Some(&rec_idx), Some(&p_rec_idx)) = (idx_map.get(i), idx_map.get(p))
                && self.records[rec_idx].parent.is_none()
            {
                self.records[rec_idx].parent = Some(p_rec_idx);
            }
        }
    }

    /// Render compact failure memory, cross-branch contrast, and sub-metric leaders for `parent`.
    pub fn render_context(&self, parent: &Candidate, component: &str) -> String {
        let p_idx = self.find_idx(parent);
        if self.records.iter().enumerate().all(|(i, _)| Some(i) == p_idx) {
            return String::new();
        }
        let parent_text = parent.get(component).map(String::as_str).unwrap_or("");
        let (parent_score, parent_objs) = p_idx
            .map(|i| (self.records[i].score, &self.records[i].objectives))
            .unwrap_or((f64::NEG_INFINITY, const { &BTreeMap::new() }));

        let mut out = String::new();

        // 1. Trial History & Rejected Edits (up to 4 rejected children + up to 4 recent others)
        let mut shown = BTreeSet::new();
        let mut hist_items = Vec::new();
        if let Some(pi) = p_idx {
            for idx in self.recent.iter().rev().copied() {
                if hist_items.len() >= 4 {
                    break;
                }
                let r = &self.records[idx];
                if r.parent == Some(pi) && !r.accepted && shown.insert(idx) {
                    hist_items.push((idx, "rejected child"));
                }
            }
        }
        let mut other_count = 0usize;
        for idx in self.recent.iter().rev().copied() {
            if other_count >= 4 {
                break;
            }
            if Some(idx) == p_idx || !shown.insert(idx) {
                continue;
            }
            let r = &self.records[idx];
            if r.texts.get(component).map(String::as_str).unwrap_or("") != parent_text {
                hist_items.push((idx, if r.accepted { "accepted" } else { "rejected" }));
                other_count += 1;
            }
        }
        if !hist_items.is_empty() {
            out.push_str("### Trial History & Rejected Edits\n");
            out.push_str("Do not repeat edits that failed or regressed above.\n");
            for (idx, tag) in hist_items {
                let r = &self.records[idx];
                let other = r.texts.get(component).map(String::as_str).unwrap_or("");
                let diff = compact_diff(parent_text, other, 30);
                let _ = writeln!(
                    out,
                    "- Candidate #{idx} ({tag}, score={:.4}, status={}):\n```diff\n{diff}\n```",
                    r.score, r.status
                );
            }
        }

        // 2. Best Accepted Branch (highest accepted score beating parent with different component text)
        let best_branch = self
            .records
            .iter()
            .enumerate()
            .filter(|(i, r)| {
                Some(*i) != p_idx
                    && r.accepted
                    && r.score > parent_score
                    && r.texts.get(component).map(String::as_str).unwrap_or("") != parent_text
            })
            .max_by(|(_, a), (_, b)| a.score.total_cmp(&b.score))
            .map(|(i, _)| i);

        if let Some(bi) = best_branch {
            let b = &self.records[bi];
            let diff = compact_diff(
                parent_text,
                b.texts.get(component).map(String::as_str).unwrap_or(""),
                40,
            );
            let _ = writeln!(
                out,
                "\n### Best Accepted Branch\n- Candidate #{bi} (score={:.4} vs current={:.4}):\n```diff\n{diff}\n```",
                b.score,
                if p_idx.is_some() { parent_score } else { 0.0 }
            );
        }

        // 3. Sub-Metric Leaders (only metrics where another candidate beats parent)
        let metric_keys: BTreeSet<&str> = self
            .records
            .iter()
            .flat_map(|r| r.objectives.keys().map(String::as_str))
            .collect();
        let mut sub_lines = Vec::new();
        for key in metric_keys {
            let cur = parent_objs.get(key).copied().unwrap_or(f64::NEG_INFINITY);
            let leader = self
                .records
                .iter()
                .enumerate()
                .filter_map(|(i, r)| r.objectives.get(key).map(|&v| (i, v)))
                .max_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((li, best_val)) = leader
                && Some(li) != p_idx
                && best_val > cur
            {
                let cur_str = if cur.is_finite() { format!("{cur:.4}") } else { "n/a".into() };
                let lead_text = self.records[li]
                    .texts
                    .get(component)
                    .map(String::as_str)
                    .unwrap_or("");
                if Some(li) == best_branch {
                    sub_lines.push(format!(
                        "- {key}: current={cur_str}, best={best_val:.4} (Candidate #{li}, same as Best Accepted Branch above)"
                    ));
                } else if lead_text == parent_text {
                    sub_lines.push(format!(
                        "- {key}: current={cur_str}, best={best_val:.4} (Candidate #{li}, unchanged in `{component}`)"
                    ));
                } else {
                    let diff = compact_diff(parent_text, lead_text, 30);
                    sub_lines.push(format!(
                        "- {key}: current={cur_str}, best={best_val:.4} (Candidate #{li}):\n```diff\n{diff}\n```"
                    ));
                }
            }
        }
        if !sub_lines.is_empty() {
            out.push_str("\n### Sub-Metric Leaders\n");
            for line in sub_lines {
                out.push_str(&line);
                out.push('\n');
            }
        }

        truncate_utf8(out.trim(), 12_000).to_string()
    }
}

fn mean_score(scores: &[f64]) -> f64 {
    if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f64>() / scores.len() as f64
    }
}

fn mean_objectives(per_item: Option<&[BTreeMap<String, f64>]>) -> BTreeMap<String, f64> {
    let Some(items) = per_item else {
        return BTreeMap::new();
    };
    let mut acc: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for map in items {
        for (k, &v) in map {
            let e = acc.entry(k.clone()).or_default();
            e.0 += v;
            e.1 += 1;
        }
    }
    acc.into_iter()
        .map(|(k, (sum, n))| (k, sum / n.max(1) as f64))
        .collect()
}

fn summarize_status(feedback: &[Value]) -> String {
    for item in feedback {
        let Some(obj) = item.as_object() else {
            continue;
        };
        for key in ["violation", "error", "Detail"] {
            if let Some(s) = obj.get(key).and_then(Value::as_str)
                && !s.starts_with("PASS ")
            {
                return first_line_trimmed(s, 180);
            }
        }
        if let Some(stdout) = obj.get("stdout").and_then(Value::as_str)
            && let Some(line) = stdout.lines().find(|l| l.starts_with("FAILED"))
        {
            return first_line_trimmed(line, 180);
        }
        if let Some(code) = obj.get("exit_code").and_then(Value::as_i64)
            && code != 0
        {
            return format!("exit={code}");
        }
        if obj.get("timed_out").and_then(Value::as_bool) == Some(true) {
            return "timeout".to_string();
        }
    }
    "ok".to_string()
}

fn first_line_trimmed(s: &str, max_bytes: usize) -> String {
    let line = s.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
    truncate_utf8(line, max_bytes).to_string()
}

fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Compact line diff with common-prefix/suffix trimming and bounded LCS hunks.
pub(crate) fn compact_diff(old: &str, new: &str, max_lines: usize) -> String {
    if old == new {
        return "(unchanged in this component)".to_string();
    }
    let a: Vec<&str> = old.lines().collect();
    let b: Vec<&str> = new.lines().collect();

    let mut pre = 0;
    while pre < a.len() && pre < b.len() && a[pre] == b[pre] {
        pre += 1;
    }
    let mut suf = 0;
    while suf < a.len() - pre && suf < b.len() - pre && a[a.len() - 1 - suf] == b[b.len() - 1 - suf] {
        suf += 1;
    }

    let a_mid = &a[pre..a.len() - suf];
    let b_mid = &b[pre..b.len() - suf];
    let mut diff_lines = Vec::new();
    if pre > 0 {
        diff_lines.push(format!(" {}", a[pre - 1]));
    }

    // shortcut: O(N*M) LCS capped at 250 changed lines per side, upgrade to Myers diff if huge full-file diffs become common
    if a_mid.len() <= 250 && b_mid.len() <= 250 {
        let (m, n) = (a_mid.len(), b_mid.len());
        let w = n + 1;
        let mut dp = vec![0u16; (m + 1) * w];
        for i in (0..m).rev() {
            for j in (0..n).rev() {
                dp[i * w + j] = if a_mid[i] == b_mid[j] {
                    dp[(i + 1) * w + (j + 1)].saturating_add(1)
                } else {
                    dp[(i + 1) * w + j].max(dp[i * w + (j + 1)])
                };
            }
        }
        let (mut i, mut j) = (0, 0);
        while i < m && j < n {
            if a_mid[i] == b_mid[j] {
                diff_lines.push(format!(" {}", a_mid[i]));
                i += 1;
                j += 1;
            } else if dp[(i + 1) * w + j] >= dp[i * w + (j + 1)] {
                diff_lines.push(format!("-{}", a_mid[i]));
                i += 1;
            } else {
                diff_lines.push(format!("+{}", b_mid[j]));
                j += 1;
            }
        }
        while i < m {
            diff_lines.push(format!("-{}", a_mid[i]));
            i += 1;
        }
        while j < n {
            diff_lines.push(format!("+{}", b_mid[j]));
            j += 1;
        }
    } else {
        for line in a_mid {
            diff_lines.push(format!("-{line}"));
        }
        for line in b_mid {
            diff_lines.push(format!("+{line}"));
        }
    }

    if suf > 0 {
        diff_lines.push(format!(" {}", a[a.len() - suf]));
    }

    if diff_lines.len() > max_lines {
        let extra = diff_lines.len() - max_lines;
        diff_lines.truncate(max_lines);
        diff_lines.push(format!("... (+{extra} more diff lines)"));
    }
    diff_lines.join("\n")
}
