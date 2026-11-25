use std::collections::{BTreeMap, BTreeSet};

use gepa_core::merge::{self, Merge, MergeConfig};
use gepa_core::pareto::remove_dominated;
use gepa_core::rng::Rng;
use gepa_core::state::{FrontierKey, ValEval};
use gepa_core::strategies::Stop;
use gepa_core::{Adapter, Candidate, Config, DataId, Engine, EvalBatch, Event, Split, State};
use serde_json::json;

const LETTERS: &[u8] = b"abcdefgh";

/// Example k scores 1 when the prompt contains letter k; the proposer appends a missed letter.
#[derive(Default)]
struct Toy {
    events: Vec<Event>,
}

impl Adapter for Toy {
    fn evaluate(&mut self, c: &Candidate, _: Split, ids: &[DataId], traces: bool) -> gepa_core::Result<EvalBatch> {
        let text = &c["prompt"];
        let hit = |k: usize| text.contains(LETTERS[k] as char);
        Ok(EvalBatch {
            scores: ids.iter().map(|&k| hit(k) as u8 as f64).collect(),
            objectives: Some(ids.iter().map(|&k| BTreeMap::from([("len".into(), -(text.len() as f64)), ("hit".into(), hit(k) as u8 as f64)])).collect()),
            feedback: if traces { ids.iter().map(|&k| json!({"want": (LETTERS[k] as char).to_string()})).collect() } else { vec![] },
            metric_calls: None,
        })
    }

    fn propose(&mut self, parent: &Candidate, _: &[DataId], eval: &EvalBatch, comps: &[String]) -> gepa_core::Result<BTreeMap<String, String>> {
        let text = &parent[&comps[0]];
        let missed = eval.feedback.iter().zip(&eval.scores).find(|(_, s)| **s < 1.0).map(|(f, _)| f["want"].as_str().unwrap().to_string());
        Ok(BTreeMap::from([(comps[0].clone(), format!("{text}{}", missed.unwrap_or_default()))]))
    }

    fn on_event(&mut self, e: &Event) {
        self.events.push(e.clone());
    }
}

fn config() -> Config {
    Config {
        stop: Stop { max_metric_calls: Some(400), score_threshold: Some(1.0), ..Stop::default() },
        perfect_score: Some(1.0),
        skip_perfect_score: true,
        merge: Some(MergeConfig::default()),
        ..Config::default()
    }
}

fn run() -> (Engine, Toy) {
    let mut toy = Toy::default();
    let seed = Candidate::from([("prompt".into(), "x".into())]);
    let mut engine = Engine::new(config(), seed, 8, 8, &mut toy).unwrap();
    engine.run(&mut toy).unwrap();
    (engine, toy)
}

#[test]
fn optimizes_toy_task_deterministically_and_round_trips() {
    let (engine, toy) = run();
    let (best, cand) = engine.best();
    assert_eq!(engine.state.val_average(best), 1.0, "best: {cand:?}");
    assert!(engine.state.total_evals <= 400 + 8 * 3, "budget overshoot {}", engine.state.total_evals);
    assert!(toy.events.iter().any(|e| matches!(e, Event::NewCandidate { .. })));

    // Same seed -> identical run; checkpoint JSON restores it exactly.
    let a = serde_json::to_string(&engine).unwrap();
    assert_eq!(a, serde_json::to_string(&run().0).unwrap());
    let back: Engine = serde_json::from_str(&a).unwrap();
    assert_eq!(a, serde_json::to_string(&back).unwrap());
}

#[test]
fn no_stop_condition_is_rejected() {
    let seed = Candidate::from([("prompt".into(), "x".into())]);
    assert!(Engine::new(Config::default(), seed, 8, 8, &mut Toy::default()).is_err());
}

#[test]
fn domination_pruning_keeps_unique_front_holders() {
    // 0 is on every front 1 is on, and also alone on front C -> 1 is dominated.
    let fronts = BTreeMap::from([
        (FrontierKey::Val(0), BTreeSet::from([0, 1])),
        (FrontierKey::Val(1), BTreeSet::from([0, 1, 2])),
        (FrontierKey::Val(2), BTreeSet::from([0])),
        (FrontierKey::Val(3), BTreeSet::from([2])),
    ]);
    let pruned = remove_dominated(&fronts, &[0.9, 0.5, 0.4]);
    let left: BTreeSet<usize> = pruned.into_values().flatten().collect();
    assert_eq!(left, BTreeSet::from([0, 2]));
}

#[test]
fn merge_combines_disjoint_edits() {
    let val = |s: f64| ValEval { scores: (0..5).map(|i| (i, s)).collect(), objectives: BTreeMap::new() };
    let cand = |a: &str, b: &str| Candidate::from([("a".into(), a.into()), ("b".into(), b.into())]);
    let mut state = State::new(cand("a0", "b0"), &val(0.1), Default::default());
    state.add(vec![0], cand("a1", "b0"), &val(0.5), 0);
    state.add(vec![0], cand("a0", "b1"), &val(0.6), 0);
    let mut m = Merge::new(MergeConfig::default());
    let scores = state.val_averages();
    let (child, i, j, anc) = merge::attempt(&state, &scores, &[1, 2], &mut m, &mut Rng::new(0)).unwrap();
    assert_eq!((i, j, anc), (1, 2, 0));
    assert_eq!(child, cand("a1", "b1"));
    // The same merge is never proposed twice.
    assert!(merge::attempt(&state, &scores, &[1, 2], &mut m, &mut Rng::new(0)).is_none());
}
