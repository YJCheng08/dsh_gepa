//! Port of `gepa.proposer.merge`: combine two descendants of a common ancestor,
//! taking each component from whichever descendant changed it.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::rng::Rng;
use crate::state::{Candidate, DataId, State};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct MergeConfig {
    /// Accepted merges allowed per run.
    pub max_invocations: usize,
    /// Minimum val ids both parents were scored on.
    pub val_overlap_floor: usize,
}

impl Default for MergeConfig {
    fn default() -> Self {
        MergeConfig { max_invocations: 5, val_overlap_floor: 5 }
    }
}

/// Merge scheduling and history (gepa `MergeProposer` counters).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Merge {
    pub config: MergeConfig,
    pub due: usize,
    pub tested: usize,
    pub last_iter_found_new: bool,
    /// (parent1, parent2, ancestor) already tried.
    pub triplets: Vec<(usize, usize, usize)>,
    /// (parent1, parent2, source candidate per component) already produced.
    pub descriptions: Vec<(usize, usize, Vec<usize>)>,
}

impl Merge {
    pub fn new(config: MergeConfig) -> Self {
        Merge { config, due: 0, tested: 0, last_iter_found_new: false, triplets: vec![], descriptions: vec![] }
    }
}

fn ancestors(parents: &[Vec<usize>], node: usize) -> BTreeSet<usize> {
    let mut found = BTreeSet::new();
    let mut stack = parents[node].clone();
    while let Some(p) = stack.pop() {
        if found.insert(p) {
            stack.extend(&parents[p]);
        }
    }
    found
}

/// Some component is unchanged in exactly one descendant, so a merge combines two edits.
fn desirable(c: &[Candidate], anc: usize, i: usize, j: usize) -> bool {
    c[anc].iter().any(|(name, a)| {
        let (x, y) = (&c[i][name], &c[j][name]);
        (a == x || a == y) && x != y
    })
}

fn common_ancestor_pair(state: &State, scores: &[f64], pool: &[usize], m: &Merge, rng: &mut Rng) -> Option<(usize, usize, usize)> {
    for _ in 0..10 {
        let pair = rng.sample(pool, 2);
        let (i, j) = (pair[0].min(pair[1]), pair[0].max(pair[1]));
        let (ai, aj) = (ancestors(&state.parents, i), ancestors(&state.parents, j));
        if ai.contains(&j) || aj.contains(&i) {
            continue;
        }
        let common: Vec<usize> = ai
            .intersection(&aj)
            .copied()
            .filter(|&a| {
                !m.triplets.contains(&(i, j, a))
                    && scores[a] <= scores[i]
                    && scores[a] <= scores[j]
                    && desirable(&state.candidates, a, i, j)
            })
            .collect();
        if !common.is_empty() {
            let weights: Vec<f64> = common.iter().map(|&a| scores[a]).collect();
            return Some((i, j, common[rng.weighted(&weights)]));
        }
    }
    None
}

/// A new merged candidate from non-dominated `pool`, as `(child, parent1, parent2, ancestor)`.
/// Records the description so the same merge is never produced twice.
pub fn attempt(state: &State, scores: &[f64], pool: &[usize], m: &mut Merge, rng: &mut Rng) -> Option<(Candidate, usize, usize, usize)> {
    if pool.len() < 2 || state.candidates.len() < 3 {
        return None;
    }
    for _ in 0..10 {
        let Some((i, j, anc)) = common_ancestor_pair(state, scores, pool, m, rng) else { continue };
        let c = &state.candidates;
        let mut child = c[anc].clone();
        let mut desc = Vec::new();
        for (name, a) in &c[anc] {
            let (x, y) = (&c[i][name], &c[j][name]);
            let from = if (a == x || a == y) && x != y {
                if a == x { j } else { i }
            } else if a != x && a != y {
                // both changed it: take the better descendant's
                if scores[i] > scores[j] {
                    i
                } else if scores[j] > scores[i] {
                    j
                } else {
                    rng.choice(&[i, j])
                }
            } else {
                i // x == y
            };
            child.insert(name.clone(), c[from][name].clone());
            desc.push(from);
        }
        if m.descriptions.contains(&(i, j, desc.clone())) {
            continue;
        }
        let overlap = state.val_scores[i].keys().filter(|k| state.val_scores[j].contains_key(k)).count();
        if overlap < m.config.val_overlap_floor {
            continue;
        }
        m.descriptions.push((i, j, desc));
        return Some((child, i, j, anc));
    }
    None
}

/// Up to `n` shared val ids, spread over "1 wins / 2 wins / tie" so the merge is
/// tested where its parents differ.
pub fn subsample(s1: &BTreeMap<DataId, f64>, s2: &BTreeMap<DataId, f64>, n: usize, rng: &mut Rng) -> Vec<DataId> {
    let common: Vec<DataId> = s1.keys().copied().filter(|k| s2.contains_key(k)).collect();
    let bucket = |f: &dyn Fn(f64, f64) -> bool| common.iter().copied().filter(|k| f(s1[k], s2[k])).collect::<Vec<_>>();
    let buckets = [bucket(&|a, b| a > b), bucket(&|a, b| b > a), bucket(&|a, b| !(a > b || b > a))];
    let each = n.div_ceil(3).max(1);
    let mut picked: Vec<DataId> = Vec::new();
    for b in &buckets {
        if picked.len() >= n {
            break;
        }
        let avail: Vec<DataId> = b.iter().copied().filter(|k| !picked.contains(k)).collect();
        let take = avail.len().min(each).min(n - picked.len());
        picked.extend(rng.sample(&avail, take));
    }
    let rest = n - picked.len();
    if rest > 0 {
        let unused: Vec<DataId> = common.iter().copied().filter(|k| !picked.contains(k)).collect();
        if unused.len() >= rest {
            picked.extend(rng.sample(&unused, rest));
        } else if !common.is_empty() {
            picked.extend((0..rest).map(|_| rng.choice(&common)));
        }
    }
    picked
}
