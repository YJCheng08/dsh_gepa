//! Port of `gepa.core.engine` + `ReflectiveMutationProposer`: the optimization loop.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::merge::{self, Merge, MergeConfig};
use crate::pareto;
use crate::rng::Rng;
use crate::state::{Candidate, DataId, FrontierType, Objectives, State, ValEval};
use crate::strategies::{Acceptance, CandidateSelector, ComponentSelector, EpochSampler, Sampling, Selection, Stop};
use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    Val,
}

/// An adapter's result for `ids`, index-aligned with them.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct EvalBatch {
    pub scores: Vec<f64>,
    /// Per-example metric scores; each metric gets its own Pareto front.
    pub objectives: Option<Vec<Objectives>>,
    /// Per-example side info for reflection. Empty = nothing to reflect on (task skipped).
    pub feedback: Vec<Value>,
    /// Budget actually spent; defaults to `scores.len()`.
    pub metric_calls: Option<usize>,
}

/// Everything host-specific: running a candidate and writing new texts.
pub trait Adapter {
    /// Score `candidate` on `ids` of `split`. `capture_traces` asks for `feedback`.
    fn evaluate(&mut self, candidate: &Candidate, split: Split, ids: &[DataId], capture_traces: bool) -> Result<EvalBatch>;
    /// New texts for `components` of `parent`, from its run `eval` on train `ids`.
    fn propose(&mut self, parent: &Candidate, ids: &[DataId], eval: &EvalBatch, components: &[String]) -> Result<BTreeMap<String, String>>;
    fn on_event(&mut self, _event: &Event) {}
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    IterationStart { iteration: usize, total_evals: usize },
    NewCandidate { iteration: usize, idx: usize, parents: Vec<usize>, val_score: f64, is_best: bool },
    Rejected { iteration: usize, parents: Vec<usize>, before: f64, after: f64, reason: String },
    Skipped { iteration: usize, parent: usize, reason: String },
    Error { iteration: usize, message: String },
}

/// Defaults follow gepa `optimize_anything` (hybrid frontier, Pareto selection, minibatch 3).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub seed: u64,
    pub minibatch_size: usize,
    pub frontier: FrontierType,
    pub candidate_selector: CandidateSelector,
    pub component_selector: ComponentSelector,
    pub sampling: Sampling,
    pub selection: Selection,
    pub acceptance: Acceptance,
    pub perfect_score: Option<f64>,
    pub skip_perfect_score: bool,
    /// `Some` enables merge.
    pub merge: Option<MergeConfig>,
    pub stop: Stop,
    /// false: report a failed iteration as `Event::Error` and keep going.
    pub raise_on_error: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            seed: 0,
            minibatch_size: 3,
            frontier: FrontierType::default(),
            candidate_selector: CandidateSelector::default(),
            component_selector: ComponentSelector::default(),
            sampling: Sampling::default(),
            selection: Selection::default(),
            acceptance: Acceptance::default(),
            perfect_score: None,
            skip_perfect_score: false,
            merge: None,
            stop: Stop::default(),
            raise_on_error: true,
        }
    }
}

/// A whole run. Serialize it between steps to checkpoint; deserialize to resume.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Engine {
    pub config: Config,
    pub state: State,
    pub train_len: usize,
    pub val_len: usize,
    rng: Rng,
    sampler: EpochSampler,
    merge: Option<Merge>,
    best_seen: Option<f64>,
    stale: usize,
    #[serde(skip)]
    started: Option<Instant>,
}

impl Engine {
    /// Score `seed` on the full valset and start a run over `train_len`/`val_len` examples.
    pub fn new(config: Config, seed: Candidate, train_len: usize, val_len: usize, adapter: &mut dyn Adapter) -> Result<Self> {
        if config.stop.is_unbounded() {
            return Err("no stop condition: set stop.max_metric_calls or another limit".into());
        }
        if seed.is_empty() || train_len == 0 || val_len == 0 || config.minibatch_size == 0 {
            return Err("seed candidate, trainset, valset and minibatch_size must be non-empty".into());
        }
        let (eval, calls) = val_eval(adapter, &seed, val_len)?;
        let mut state = State::new(seed, &eval, config.frontier);
        state.total_evals = calls;
        Ok(Engine {
            rng: Rng::new(config.seed),
            merge: config.merge.clone().map(Merge::new),
            config,
            state,
            train_len,
            val_len,
            sampler: EpochSampler::default(),
            best_seen: None,
            stale: 0,
            started: None,
        })
    }

    /// Index and text of the best candidate by valset mean.
    pub fn best(&self) -> (usize, &Candidate) {
        let i = self.state.best_idx();
        (i, &self.state.candidates[i])
    }

    /// Check limits. Call once per iteration: it advances the stale-iteration counter.
    pub fn should_stop(&mut self) -> bool {
        let started = *self.started.get_or_insert_with(Instant::now);
        let best = self.state.val_averages().into_iter().fold(f64::NEG_INFINITY, f64::max);
        if self.best_seen.is_none_or(|b| best > b) {
            self.best_seen = Some(best);
            self.stale = 0;
        } else {
            self.stale += 1;
        }
        let (s, st) = (&self.config.stop, &self.state);
        s.max_metric_calls.is_some_and(|m| st.total_evals >= m)
            || s.max_iterations.is_some_and(|m| st.iteration >= m)
            || s.max_candidates.is_some_and(|m| st.candidates.len() >= m)
            || s.score_threshold.is_some_and(|t| best >= t)
            || s.max_stale_iterations.is_some_and(|m| self.stale >= m)
            || s.timeout_secs.is_some_and(|t| started.elapsed().as_secs_f64() > t)
            || s.stop_file.as_ref().is_some_and(|p| p.exists())
    }

    /// Step until a stop condition holds. Hosts that checkpoint or cancel loop over
    /// `should_stop` + `step` themselves.
    pub fn run(&mut self, adapter: &mut dyn Adapter) -> Result<()> {
        while !self.should_stop() {
            if let Err(e) = self.step(adapter) {
                if self.config.raise_on_error {
                    return Err(e);
                }
                adapter.on_event(&Event::Error { iteration: self.state.iteration, message: e.to_string() });
            }
        }
        Ok(())
    }

    /// One iteration: a scheduled merge if one applies, else reflective mutation.
    /// Returns whether a candidate was added.
    pub fn step(&mut self, adapter: &mut dyn Adapter) -> Result<bool> {
        self.state.iteration += 1;
        adapter.on_event(&Event::IterationStart { iteration: self.state.iteration, total_evals: self.state.total_evals });
        if let Some(m) = self.merge.as_mut() {
            let due = m.due > 0 && m.last_iter_found_new;
            m.last_iter_found_new = false;
            if due {
                if let Some(added) = self.try_merge(adapter)? {
                    return Ok(added);
                }
            }
        }
        self.reflect(adapter)
    }

    fn reflect(&mut self, adapter: &mut dyn Adapter) -> Result<bool> {
        let iteration = self.state.iteration;
        let Sampling { parents, per_parent } = self.config.sampling;
        let mut tasks = Vec::new();
        for _ in 0..parents {
            let parent = self.config.candidate_selector.select(&self.state, &mut self.rng);
            for _ in 0..per_parent {
                let ids = self.sampler.next(self.train_len, self.config.minibatch_size, iteration - 1, &mut self.rng);
                tasks.push((parent, ids));
            }
        }

        // Each distinct (parent, minibatch) runs once.
        let mut parent_evals: HashMap<(usize, Vec<DataId>), EvalBatch> = HashMap::new();
        for task in &tasks {
            if !parent_evals.contains_key(task) {
                let eval = evaluate(adapter, &self.state.candidates[task.0], Split::Train, &task.1, true)?;
                self.state.total_evals += calls(&eval);
                parent_evals.insert(task.clone(), eval);
            }
        }

        let mut children = Vec::new();
        for (parent, ids) in tasks {
            let eval = &parent_evals[&(parent, ids.clone())];
            let mut skip = |reason: String| adapter.on_event(&Event::Skipped { iteration, parent, reason });
            if eval.feedback.is_empty() {
                skip("no feedback to reflect on".into());
                continue;
            }
            if self.config.skip_perfect_score && self.config.perfect_score.is_some_and(|p| eval.scores.iter().all(|s| *s >= p)) {
                skip("all minibatch scores perfect".into());
                continue;
            }
            let components = self.config.component_selector.select(&mut self.state, parent);
            let base = &self.state.candidates[parent];
            let texts = match adapter.propose(base, &ids, eval, &components) {
                Ok(t) => t,
                Err(e) => {
                    adapter.on_event(&Event::Skipped { iteration, parent, reason: format!("proposal failed: {e}") });
                    continue;
                }
            };
            let mut skip = |reason: String| adapter.on_event(&Event::Skipped { iteration, parent, reason });
            if let Some(bad) = texts.keys().find(|k| !base.contains_key(*k)) {
                skip(format!("proposal names unknown component {bad:?}"));
                continue;
            }
            let mut child = base.clone();
            child.extend(texts);
            if child == *base {
                skip("proposal changed nothing".into());
                continue;
            }
            children.push((parent, ids, child, eval.scores.iter().sum::<f64>()));
        }

        let mut proposals = Vec::new();
        for (parent, ids, child, before) in children {
            let eval = evaluate(adapter, &child, Split::Train, &ids, true)?;
            self.state.total_evals += calls(&eval);
            proposals.push((parent, child, before, eval.scores.iter().sum::<f64>()));
        }

        let sums: Vec<(f64, f64)> = proposals.iter().map(|p| (p.2, p.3)).collect();
        let selected = self.config.selection.select(&sums, self.config.acceptance);
        let keep: Vec<usize> = {
            let mut seen = HashSet::new();
            selected.iter().copied().filter(|&i| seen.insert(&proposals[i].1)).collect()
        };

        let mut added = false;
        for (i, (parent, child, before, after)) in proposals.into_iter().enumerate() {
            if keep.contains(&i) {
                self.add_candidate(adapter, child, vec![parent])?;
                added = true;
                if let Some(m) = self.merge.as_mut() {
                    m.last_iter_found_new = true;
                    if m.tested < m.config.max_invocations {
                        m.due += 1;
                    }
                }
                continue;
            }
            let reason = if selected.contains(&i) {
                "duplicate of another proposal this iteration".to_string()
            } else if self.config.acceptance.accepts(before, after) {
                format!("passed acceptance but not chosen by {:?}", self.config.selection)
            } else {
                format!("subsample score {after} not better than {before}")
            };
            adapter.on_event(&Event::Rejected { iteration, parents: vec![parent], before, after, reason });
        }
        Ok(added)
    }

    /// `None` when no merge could be formed (caller falls back to reflection).
    fn try_merge(&mut self, adapter: &mut dyn Adapter) -> Result<Option<bool>> {
        let scores = self.state.val_averages();
        let pool = pareto::dominators(&self.state.fronts(), &scores);
        let m = self.merge.as_mut().expect("merge enabled");
        let Some((child, id1, id2, ancestor)) = merge::attempt(&self.state, &scores, &pool, m, &mut self.rng) else {
            return Ok(None);
        };
        m.triplets.push((id1, id2, ancestor));
        let ids = merge::subsample(&self.state.val_scores[id1], &self.state.val_scores[id2], 5, &mut self.rng);
        if ids.is_empty() {
            return Ok(None);
        }
        let sum = |c: usize| ids.iter().map(|id| self.state.val_scores[c][id]).sum::<f64>();
        let before = sum(id1).max(sum(id2));
        let eval = evaluate(adapter, &child, Split::Val, &ids, false)?;
        self.state.total_evals += calls(&eval);
        let after: f64 = eval.scores.iter().sum();
        if after >= before {
            self.add_candidate(adapter, child, vec![id1, id2])?;
            let m = self.merge.as_mut().expect("merge enabled");
            m.due -= 1;
            m.tested += 1;
            return Ok(Some(true));
        }
        let reason = format!("merge via ancestor {ancestor}: subsample {after} below best parent {before}");
        adapter.on_event(&Event::Rejected { iteration: self.state.iteration, parents: vec![id1, id2], before, after, reason });
        Ok(Some(false))
    }

    fn add_candidate(&mut self, adapter: &mut dyn Adapter, candidate: Candidate, parents: Vec<usize>) -> Result<usize> {
        let (eval, calls) = val_eval(adapter, &candidate, self.val_len)?;
        let discovered_at = self.state.total_evals;
        self.state.total_evals += calls;
        self.state.num_full_val_evals += 1;
        let idx = self.state.add(parents.clone(), candidate, &eval, discovered_at);
        adapter.on_event(&Event::NewCandidate {
            iteration: self.state.iteration,
            idx,
            parents,
            val_score: self.state.val_average(idx),
            is_best: self.state.best_idx() == idx,
        });
        Ok(idx)
    }
}

fn calls(eval: &EvalBatch) -> usize {
    eval.metric_calls.unwrap_or(eval.scores.len())
}

fn val_eval(adapter: &mut dyn Adapter, candidate: &Candidate, val_len: usize) -> Result<(ValEval, usize)> {
    let ids: Vec<DataId> = (0..val_len).collect();
    let eval = evaluate(adapter, candidate, Split::Val, &ids, false)?;
    let objectives = eval.objectives.clone().map(|o| ids.iter().copied().zip(o).collect()).unwrap_or_default();
    let scores = ids.iter().copied().zip(eval.scores.iter().copied()).collect();
    Ok((ValEval { scores, objectives }, calls(&eval)))
}

/// `adapter.evaluate`, rejecting misaligned or non-finite results (foreign code boundary).
fn evaluate(adapter: &mut dyn Adapter, candidate: &Candidate, split: Split, ids: &[DataId], traces: bool) -> Result<EvalBatch> {
    let eval = adapter.evaluate(candidate, split, ids, traces)?;
    let n = ids.len();
    if eval.scores.len() != n
        || eval.objectives.as_ref().is_some_and(|o| o.len() != n)
        || !(eval.feedback.is_empty() || eval.feedback.len() == n)
    {
        return Err(format!("adapter returned {} scores / {} feedback for {n} examples", eval.scores.len(), eval.feedback.len()).into());
    }
    if eval.scores.iter().chain(eval.objectives.iter().flatten().flat_map(|o| o.values())).any(|s| !s.is_finite()) {
        return Err("adapter returned a non-finite score".into());
    }
    Ok(eval)
}
