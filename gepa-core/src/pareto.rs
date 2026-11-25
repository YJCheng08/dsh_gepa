//! Port of `gepa_utils`: domination pruning and frontier sampling.

use std::collections::BTreeSet;

use crate::rng::Rng;
use crate::state::Fronts;

/// Drop candidates whose every front is also held by another survivor, weakest
/// (by `scores`) first. A front's sole holder always survives.
pub fn remove_dominated(fronts: &Fronts, scores: &[f64]) -> Fronts {
    let mut programs: Vec<usize> = fronts.values().flatten().copied().collect::<BTreeSet<_>>().into_iter().collect();
    programs.sort_by(|a, b| scores[*a].total_cmp(&scores[*b]));
    let mut dominated = BTreeSet::new();
    while let Some(y) = programs.iter().copied().find(|&y| {
        !dominated.contains(&y)
            && fronts
                .values()
                .filter(|f| f.contains(&y))
                .all(|f| f.iter().any(|p| *p != y && !dominated.contains(p)))
    }) {
        dominated.insert(y);
    }
    fronts.iter().map(|(k, f)| (k.clone(), f.difference(&dominated).copied().collect())).collect()
}

/// Non-dominated candidates (merge parents).
pub fn dominators(fronts: &Fronts, scores: &[f64]) -> Vec<usize> {
    remove_dominated(fronts, scores).into_values().flatten().collect::<BTreeSet<_>>().into_iter().collect()
}

/// A candidate drawn proportionally to how many pruned fronts it holds; the best
/// aggregate when every front is empty.
pub fn sample_front(fronts: &Fronts, scores: &[f64], rng: &mut Rng) -> usize {
    let pool: Vec<usize> = remove_dominated(fronts, scores).into_values().flatten().collect();
    if pool.is_empty() { argmax(scores) } else { rng.choice(&pool) }
}

/// First index of the maximum (0 for an empty slice).
pub fn argmax(xs: &[f64]) -> usize {
    (0..xs.len()).fold(0, |b, i| if xs[i] > xs[b] { i } else { b })
}
