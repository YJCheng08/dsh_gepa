use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Index into the train or val set (the host owns the data).
pub type DataId = usize;
/// Component name -> text. A BTreeMap keeps iteration and equality deterministic.
pub type Candidate = BTreeMap<String, String>;
/// Metric name -> score.
pub type Objectives = BTreeMap<String, f64>;
/// Best score seen for one frontier key, and every candidate reaching it.
pub type Front = (f64, BTreeSet<usize>);

/// What gets its own Pareto front: each val example, each metric, both, or example x metric.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontierType {
    Instance,
    Objective,
    #[default]
    Hybrid,
    Cartesian,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FrontierKey {
    Val(DataId),
    Objective(String),
    Cartesian(DataId, String),
}

/// Frontier key -> candidates on that front.
pub type Fronts = BTreeMap<FrontierKey, BTreeSet<usize>>;

/// One candidate's full-valset result.
#[derive(Clone, Debug, Default)]
pub struct ValEval {
    pub scores: BTreeMap<DataId, f64>,
    pub objectives: BTreeMap<DataId, Objectives>,
}

/// Every candidate found so far, its scores, and the fronts (gepa `GEPAState`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub candidates: Vec<Candidate>,
    pub parents: Vec<Vec<usize>>,
    pub val_scores: Vec<BTreeMap<DataId, f64>>,
    /// Per-candidate mean of each metric over the valset.
    pub objective_scores: Vec<Objectives>,
    pub frontier_type: FrontierType,
    pub front_val: BTreeMap<DataId, Front>,
    pub front_objective: BTreeMap<String, Front>,
    pub front_cartesian: BTreeMap<DataId, BTreeMap<String, Front>>,
    /// Component names in round-robin order (the seed's keys).
    pub components: Vec<String>,
    /// Per candidate: index into `components` to mutate next.
    pub next_component: Vec<usize>,
    /// Iterations started (gepa's `state.i + 1`).
    pub iteration: usize,
    /// Metric calls spent, including the seed's valset run.
    pub total_evals: usize,
    pub num_full_val_evals: usize,
    /// Per candidate: `total_evals` when it was found.
    pub discovered_at: Vec<usize>,
}

impl State {
    pub fn new(seed: Candidate, eval: &ValEval, frontier_type: FrontierType) -> Self {
        let mut state = State {
            components: seed.keys().cloned().collect(),
            candidates: vec![],
            parents: vec![],
            val_scores: vec![],
            objective_scores: vec![],
            frontier_type,
            front_val: BTreeMap::new(),
            front_objective: BTreeMap::new(),
            front_cartesian: BTreeMap::new(),
            next_component: vec![],
            iteration: 0,
            total_evals: 0,
            num_full_val_evals: 1,
            discovered_at: vec![],
        };
        state.add(vec![], seed, eval, 0);
        state
    }

    /// Append a fully evaluated candidate and update every front. Returns its index.
    pub fn add(&mut self, parents: Vec<usize>, candidate: Candidate, eval: &ValEval, discovered_at: usize) -> usize {
        let idx = self.candidates.len();
        let next = parents.iter().map(|&p| self.next_component[p]).max().unwrap_or(0);
        self.next_component.push(next);
        self.parents.push(parents);
        self.candidates.push(candidate);
        self.discovered_at.push(discovered_at);

        let objectives = mean_objectives(&eval.objectives);
        for (&id, &score) in &eval.scores {
            bump(self.front_val.entry(id), score, idx);
        }
        for (name, &score) in &objectives {
            bump(self.front_objective.entry(name.clone()), score, idx);
        }
        if self.frontier_type == FrontierType::Cartesian {
            for (&id, objs) in &eval.objectives {
                let row = self.front_cartesian.entry(id).or_default();
                for (name, &score) in objs {
                    bump(row.entry(name.clone()), score, idx);
                }
            }
        }
        self.val_scores.push(eval.scores.clone());
        self.objective_scores.push(objectives);
        idx
    }

    pub fn fronts(&self) -> Fronts {
        let val = || self.front_val.iter().map(|(id, f)| (FrontierKey::Val(*id), f.1.clone()));
        let obj = || self.front_objective.iter().map(|(n, f)| (FrontierKey::Objective(n.clone()), f.1.clone()));
        match self.frontier_type {
            FrontierType::Instance => val().collect(),
            FrontierType::Objective => obj().collect(),
            FrontierType::Hybrid => val().chain(obj()).collect(),
            FrontierType::Cartesian => self
                .front_cartesian
                .iter()
                .flat_map(|(id, row)| row.iter().map(move |(n, f)| (FrontierKey::Cartesian(*id, n.clone()), f.1.clone())))
                .collect(),
        }
    }

    /// Mean valset score; -inf when nothing was evaluated.
    pub fn val_average(&self, idx: usize) -> f64 {
        let s = &self.val_scores[idx];
        if s.is_empty() { f64::NEG_INFINITY } else { s.values().sum::<f64>() / s.len() as f64 }
    }

    pub fn val_averages(&self) -> Vec<f64> {
        (0..self.candidates.len()).map(|i| self.val_average(i)).collect()
    }

    /// Highest valset mean; ties go to wider coverage, then the earlier candidate.
    pub fn best_idx(&self) -> usize {
        let mut best: Option<(usize, f64, usize)> = None;
        for i in 0..self.candidates.len() {
            let (avg, cov) = (self.val_average(i), self.val_scores[i].len());
            if best.is_none_or(|(_, b, c)| avg > b || (avg == b && cov > c)) {
                best = Some((i, avg, cov));
            }
        }
        best.map_or(0, |b| b.0)
    }
}

fn bump<K: Ord>(entry: Entry<K, Front>, score: f64, idx: usize) {
    let front = entry.or_insert((f64::NEG_INFINITY, BTreeSet::new()));
    if score > front.0 {
        *front = (score, BTreeSet::from([idx]));
    } else if score == front.0 {
        front.1.insert(idx);
    }
}

fn mean_objectives(per_id: &BTreeMap<DataId, Objectives>) -> Objectives {
    let mut acc: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for objs in per_id.values() {
        for (name, &v) in objs {
            let e = acc.entry(name.clone()).or_default();
            e.0 += v;
            e.1 += 1;
        }
    }
    acc.into_iter().map(|(n, (total, count))| (n, total / count as f64)).collect()
}
