//! Engine runner: Adapter wiring, NDJSON event stream, atomic checkpointing, status, and stop.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use gepa_core::{Adapter, Candidate, DataId, Engine, EvalBatch, Event, Split};
use gepa_eval::{AnyEvaluator, Evaluator};
use gepa_reflect::{OpenAiClient, Reflector};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::schema::LoadedTask;

#[derive(Clone, Debug)]
pub struct RunOptions {
    pub run_dir: PathBuf,
    pub budget: Option<usize>,
    pub max_iterations: Option<usize>,
    pub lm_base_url: Option<String>,
    pub lm_model: Option<String>,
    pub concurrency: Option<usize>,
    pub fresh: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SummaryJson {
    pub best_idx: usize,
    pub best_val_score: f64,
    pub test_score: Option<f64>,
    pub total_evals: usize,
    pub iterations: usize,
    pub num_candidates: usize,
    pub frontier_counts: FrontierCounts,
    pub best_objectives: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrontierCounts {
    pub instance: usize,
    pub objective: usize,
    pub cartesian: usize,
}

pub struct CliAdapter {
    evaluator: AnyEvaluator,
    reflector: Reflector<OpenAiClient>,
    train_data: Vec<Value>,
    val_data: Vec<Value>,
    log_path: PathBuf,
}

impl CliAdapter {
    fn log_line(&self, line: &str) {
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&self.log_path) {
            let _ = writeln!(file, "{line}");
        }
    }

    fn emit_json(&self, val: &Value) {
        if let Ok(line) = serde_json::to_string(val) {
            println!("{line}");
            let _ = std::io::stdout().flush();
            self.log_line(&line);
        }
    }
}

impl Adapter for CliAdapter {
    fn evaluate(
        &mut self,
        candidate: &Candidate,
        split: Split,
        ids: &[DataId],
        capture_traces: bool,
    ) -> gepa_core::Result<EvalBatch> {
        let dataset = match split {
            Split::Train => &self.train_data,
            Split::Val => &self.val_data,
        };
        let mut examples = Vec::with_capacity(ids.len());
        for &id in ids {
            let ex = dataset
                .get(id)
                .ok_or_else(|| format!("example id {id} out of range for {split:?} (len={})", dataset.len()))?;
            examples.push(ex.clone());
        }
        self.evaluator
            .evaluate(candidate, split, ids, &examples, capture_traces)
    }

    fn propose(
        &mut self,
        parent: &Candidate,
        ids: &[DataId],
        eval: &EvalBatch,
        components: &[String],
    ) -> gepa_core::Result<BTreeMap<String, String>> {
        self.reflector.propose(parent, ids, eval, components)
    }

    fn on_event(&mut self, event: &Event) {
        if let Ok(val) = serde_json::to_value(event) {
            self.emit_json(&val);
        }
    }
}

/// Atomically write `bytes` to `path` via `<name>.tmp` in the same directory.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> gepa_core::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_name = format!(
        "{}.tmp.{}",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("gepa"),
        std::process::id()
    );
    let tmp_path = path.with_file_name(tmp_name);
    {
        let mut f = fs::File::create(&tmp_path)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp_path, path)?;
    Ok(())
}

pub fn run_task(mut task: LoadedTask, opts: RunOptions) -> gepa_core::Result<SummaryJson> {
    fs::create_dir_all(&opts.run_dir)?;
    let log_path = opts.run_dir.join("run.log");

    for w in &task.warnings {
        eprintln!("warning: {w}");
    }

    if let Some(base) = opts.lm_base_url {
        task.spec.reflector.lm.base_url = base;
    }
    if let Some(model) = opts.lm_model {
        task.spec.reflector.lm.model = model;
    }
    if let Some(conc) = opts.concurrency {
        task.spec.evaluator.set_concurrency(conc);
    }

    let stop_file = opts.run_dir.join("STOP");
    if stop_file.exists() {
        let _ = fs::remove_file(&stop_file);
    }

    let mut config = task.spec.engine.clone();
    if let Some(b) = opts.budget {
        config.stop.max_metric_calls = Some(b);
    }
    if let Some(m) = opts.max_iterations {
        config.stop.max_iterations = Some(m);
    }
    config.stop.stop_file = Some(stop_file);

    let client = OpenAiClient::new(task.spec.reflector.lm.clone());
    let reflector = Reflector::new(client, task.spec.reflector.prompt.clone());
    let evaluator = AnyEvaluator::new(task.spec.evaluator.clone());

    let mut adapter = CliAdapter {
        evaluator,
        reflector,
        train_data: task.train_data.clone(),
        val_data: task.val_data.clone(),
        log_path,
    };

    let ckpt_path = opts.run_dir.join("checkpoint.json");
    let mut engine = if !opts.fresh && ckpt_path.exists() {
        let raw = fs::read_to_string(&ckpt_path)?;
        let mut resumed: Engine = serde_json::from_str(&raw)?;
        resumed.config.stop = config.stop.clone();
        adapter.emit_json(&json!({
            "event": "resumed",
            "iteration": resumed.state.iteration,
            "total_evals": resumed.state.total_evals,
            "candidates": resumed.state.candidates.len(),
        }));
        resumed
    } else {
        let eng = Engine::new(
            config,
            task.seed.clone(),
            task.train_data.len(),
            task.val_data.len(),
            &mut adapter,
        )?;
        let seed_score = eng.state.val_average(0);
        adapter.emit_json(&json!({
            "event": "seed_evaluated",
            "val_score": seed_score,
            "total_evals": eng.state.total_evals,
        }));
        persist_checkpoint(&opts.run_dir, &eng)?;
        eng
    };

    while !engine.should_stop() {
        match engine.step(&mut adapter) {
            Ok(_) => persist_checkpoint(&opts.run_dir, &engine)?,
            Err(e) => {
                if engine.config.raise_on_error {
                    let _ = persist_checkpoint(&opts.run_dir, &engine);
                    return Err(e);
                }
                adapter.on_event(&Event::Error {
                    iteration: engine.state.iteration,
                    message: e.to_string(),
                });
                persist_checkpoint(&opts.run_dir, &engine)?;
            }
        }
    }

    persist_checkpoint(&opts.run_dir, &engine)?;

    let (best_idx, best_cand) = engine.best();
    let best_cand_owned = best_cand.clone();
    let test_score = if !task.test_data.is_empty() {
        let ids: Vec<DataId> = (0..task.test_data.len()).collect();
        let batch = adapter
            .evaluator
            .evaluate(&best_cand_owned, Split::Val, &ids, &task.test_data, false)?;
        let mean = if batch.scores.is_empty() {
            0.0
        } else {
            batch.scores.iter().sum::<f64>() / batch.scores.len() as f64
        };
        Some(mean)
    } else {
        None
    };

    let summary = build_summary(&engine, best_idx, test_score);
    write_atomic(
        &opts.run_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary)?.as_bytes(),
    )?;
    adapter.emit_json(&json!({
        "event": "completed",
        "best_idx": summary.best_idx,
        "best_val_score": summary.best_val_score,
        "test_score": summary.test_score,
        "total_evals": summary.total_evals,
        "iterations": summary.iterations,
        "num_candidates": summary.num_candidates,
    }));

    Ok(summary)
}

fn persist_checkpoint(run_dir: &Path, engine: &Engine) -> gepa_core::Result<()> {
    write_atomic(&run_dir.join("checkpoint.json"), serde_json::to_string(engine)?.as_bytes())?;
    let (_, best) = engine.best();
    write_atomic(
        &run_dir.join("best_candidate.json"),
        serde_json::to_string_pretty(best)?.as_bytes(),
    )?;
    let best_dir = run_dir.join("best");
    fs::create_dir_all(&best_dir)?;
    for (comp, text) in best {
        let fname = format!("{}.txt", sanitize_component_name(comp));
        write_atomic(&best_dir.join(fname), text.as_bytes())?;
    }
    Ok(())
}

fn sanitize_component_name(name: &str) -> String {
    let clean: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    if clean.is_empty() { "component".to_string() } else { clean }
}

fn build_summary(engine: &Engine, best_idx: usize, test_score: Option<f64>) -> SummaryJson {
    SummaryJson {
        best_idx,
        best_val_score: engine.state.val_average(best_idx),
        test_score,
        total_evals: engine.state.total_evals,
        iterations: engine.state.iteration,
        num_candidates: engine.state.candidates.len(),
        frontier_counts: FrontierCounts {
            instance: engine.state.front_val.len(),
            objective: engine.state.front_objective.len(),
            cartesian: engine
                .state
                .front_cartesian
                .values()
                .map(|m| m.len())
                .sum(),
        },
        best_objectives: engine
            .state
            .objective_scores
            .get(best_idx)
            .cloned()
            .unwrap_or_default(),
    }
}

pub fn status_text(run_dir: &Path, as_json: bool) -> gepa_core::Result<String> {
    let ckpt_path = run_dir.join("checkpoint.json");
    let raw = fs::read_to_string(&ckpt_path)
        .map_err(|e| format!("failed to read `{}`: {e}", ckpt_path.display()))?;
    let engine: Engine = serde_json::from_str(&raw)?;
    let (best_idx, best_cand) = engine.best();
    let test_score = run_dir
        .join("summary.json")
        .exists()
        .then(|| fs::read_to_string(run_dir.join("summary.json")).ok())
        .flatten()
        .and_then(|s| serde_json::from_str::<SummaryJson>(&s).ok())
        .and_then(|s| s.test_score);

    let summary = build_summary(&engine, best_idx, test_score);
    if as_json {
        return Ok(serde_json::to_string_pretty(&summary)?);
    }

    let budget_str = engine
        .config
        .stop
        .max_metric_calls
        .map(|b| b.to_string())
        .unwrap_or_else(|| "∞".to_string());
    // Line 1 is intentionally self-contained for collapsed UI views.
    let mut out = format!(
        "gepa status: iter={} evals={}/{} candidates={} best=#{} (val={:.4})",
        summary.iterations,
        summary.total_evals,
        budget_str,
        summary.num_candidates,
        summary.best_idx,
        summary.best_val_score
    );
    if let Some(ts) = summary.test_score {
        out.push_str(&format!(" test={ts:.4}"));
    }
    out.push_str(&format!(
        "\nfrontiers: instance={} objective={} cartesian={}",
        summary.frontier_counts.instance,
        summary.frontier_counts.objective,
        summary.frontier_counts.cartesian
    ));
    if !summary.best_objectives.is_empty() {
        let objs: Vec<String> = summary
            .best_objectives
            .iter()
            .map(|(k, v)| format!("{k}={v:.4}"))
            .collect();
        out.push_str(&format!("\nobjectives: {}", objs.join(", ")));
    }
    out.push_str(&format!(
        "\ncomponents: {}",
        best_cand.keys().cloned().collect::<Vec<_>>().join(", ")
    ));
    Ok(out)
}

pub fn stop_run(run_dir: &Path) -> gepa_core::Result<PathBuf> {
    fs::create_dir_all(run_dir)?;
    let stop_file = run_dir.join("STOP");
    write_atomic(&stop_file, b"stop\n")?;
    Ok(stop_file)
}
