//! Port of `gepa.strategies` and `gepa.utils.stop_condition`. Every strategy is a
//! serializable enum/struct so hosts configure it from JSON.

use std::collections::BTreeSet;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::pareto::{argmax, sample_front};
use crate::rng::Rng;
use crate::state::{DataId, Fronts, State};

/// Which candidate to mutate next.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CandidateSelector {
    #[default]
    Pareto,
    CurrentBest,
    EpsilonGreedy { epsilon: f64 },
    /// Pareto sampling restricted to the top `k` by valset mean.
    TopKPareto { k: usize },
}

impl CandidateSelector {
    pub fn select(&self, state: &State, rng: &mut Rng) -> usize {
        let scores = state.val_averages();
        match *self {
            Self::Pareto => sample_front(&state.fronts(), &scores, rng),
            Self::CurrentBest => argmax(&scores),
            Self::EpsilonGreedy { epsilon } => {
                if rng.f64() < epsilon { rng.below(scores.len()) } else { argmax(&scores) }
            }
            Self::TopKPareto { k } => {
                let mut order: Vec<usize> = (0..scores.len()).collect();
                order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));
                let top: BTreeSet<usize> = order.into_iter().take(k).collect();
                let fronts: Fronts = state
                    .fronts()
                    .into_iter()
                    .map(|(key, f)| (key, &f & &top))
                    .filter(|(_, f)| !f.is_empty())
                    .collect();
                if fronts.is_empty() { argmax(&scores) } else { sample_front(&fronts, &scores, rng) }
            }
        }
    }
}

/// Which components of the parent the proposer rewrites.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentSelector {
    #[default]
    RoundRobin,
    All,
}

impl ComponentSelector {
    pub fn select(self, state: &mut State, parent: usize) -> Vec<String> {
        match self {
            Self::All => state.candidates[parent].keys().cloned().collect(),
            Self::RoundRobin => {
                let pid = state.next_component[parent];
                state.next_component[parent] = (pid + 1) % state.components.len();
                vec![state.components[pid].clone()]
            }
        }
    }
}

/// Gate on minibatch score sums, child vs parent.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Acceptance {
    #[default]
    StrictImprovement,
    ImprovementOrEqual,
}

impl Acceptance {
    pub fn accepts(self, before: f64, after: f64) -> bool {
        match self {
            Self::StrictImprovement => after > before,
            Self::ImprovementOrEqual => after >= before,
        }
    }
}

/// Which accepted proposals of one iteration get a full valset run.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Selection {
    #[default]
    All,
    /// The single largest gain.
    Best,
    TopK { k: usize },
}

impl Selection {
    /// Indices into `sums` (`(before, after)` minibatch sums) to keep.
    pub fn select(&self, sums: &[(f64, f64)], acceptance: Acceptance) -> Vec<usize> {
        let mut passing: Vec<usize> = (0..sums.len()).filter(|&i| acceptance.accepts(sums[i].0, sums[i].1)).collect();
        let gain = |i: usize| sums[i].1 - sums[i].0;
        match *self {
            Self::All => passing,
            Self::Best => passing.into_iter().reduce(|b, i| if gain(i) > gain(b) { i } else { b }).into_iter().collect(),
            Self::TopK { k } => {
                passing.sort_by(|a, b| gain(*b).total_cmp(&gain(*a)));
                passing.truncate(k);
                passing
            }
        }
    }
}

/// Proposals per iteration: `parents` x `per_parent` minibatches. (1,1) is classic GEPA,
/// (1,n) gepa's SameParentSampling, (n,1) IndependentSampling, (p,n) PxNSampling.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Sampling {
    pub parents: usize,
    pub per_parent: usize,
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling { parents: 1, per_parent: 1 }
    }
}

/// gepa `EpochShuffledBatchSampler`: reshuffle each epoch, pad the last minibatch with
/// the least-used ids; repeat calls within one iteration take the following chunks.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EpochSampler {
    shuffled: Vec<DataId>,
    epoch: Option<usize>,
    last_len: usize,
    current_iteration: Option<usize>,
    calls: usize,
}

impl EpochSampler {
    /// Minibatch of `size` train ids out of `len` for 0-based iteration `i`.
    pub fn next(&mut self, len: usize, size: usize, i: usize, rng: &mut Rng) -> Vec<DataId> {
        assert!(len > 0 && size > 0, "empty trainset or minibatch");
        if self.current_iteration == Some(i) {
            self.calls += 1;
        } else {
            self.current_iteration = Some(i);
            self.calls = 0;
        }
        let base = i * size;
        let epoch = if self.epoch.is_none() { 0 } else { base / self.shuffled.len().max(1) };
        if self.shuffled.is_empty() || len != self.last_len || self.epoch.is_none_or(|e| epoch > e) {
            self.epoch = Some(epoch);
            self.last_len = len;
            self.shuffled = (0..len).collect();
            rng.shuffle(&mut self.shuffled);
            let mut freq = vec![1usize; len];
            while self.shuffled.len() % size != 0 {
                // least used id, latest in shuffle order on ties (Counter.most_common()[::-1][0])
                let id = *self.shuffled[..len].iter().rev().min_by_key(|&&id| freq[id]).unwrap();
                freq[id] += 1;
                self.shuffled.push(id);
            }
        }
        let start = (base + self.calls * size) % self.shuffled.len();
        self.shuffled[start..start + size].to_vec()
    }
}

/// Stop when any set limit is hit (gepa `CompositeStopper(mode="any")`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Stop {
    pub max_metric_calls: Option<usize>,
    pub max_iterations: Option<usize>,
    pub max_candidates: Option<usize>,
    /// Best valset mean reaching this.
    pub score_threshold: Option<f64>,
    /// Iterations in a row without a new best valset mean.
    pub max_stale_iterations: Option<usize>,
    pub timeout_secs: Option<f64>,
    /// Stop once this file exists (cross-tool graceful stop).
    pub stop_file: Option<PathBuf>,
}

impl Stop {
    pub fn is_unbounded(&self) -> bool {
        self.max_metric_calls.is_none()
            && self.max_iterations.is_none()
            && self.max_candidates.is_none()
            && self.score_threshold.is_none()
            && self.max_stale_iterations.is_none()
            && self.timeout_secs.is_none()
            && self.stop_file.is_none()
    }
}
