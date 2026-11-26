//! `gepa` CLI entrypoint: run, init, status, stop, convert.

mod legacy;
mod runner;
mod schema;

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use gepa_eval::{EvalConfig, HttpEvalConfig, JsonCommandConfig, ShellCheckConfig};

use crate::legacy::{LegacyTask, convert_legacy};
use crate::runner::{RunOptions, run_task, status_text, stop_run, write_atomic};
use crate::schema::{DataMode, DatasetSpec, ReflectorSpec, TaskSpec, default_engine_config, load_task_from_file};

#[derive(Parser)]
#[command(name = "gepa", version, about = "Cross-platform GEPA reflective Pareto optimizer")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum EvalKind {
    CommandJson,
    ShellCheck,
    Http,
}

#[derive(Subcommand)]
enum Commands {
    /// Run or resume an optimization task from a task JSON file (universal or legacy).
    Run {
        /// Path to task JSON file.
        task: PathBuf,
        /// Directory for checkpoints, logs, and best candidate outputs.
        #[arg(long)]
        run_dir: Option<PathBuf>,
        /// Override `engine.stop.max_metric_calls`.
        #[arg(long)]
        budget: Option<usize>,
        /// Override `engine.stop.max_iterations`.
        #[arg(long)]
        max_iterations: Option<usize>,
        /// Override `reflector.lm.base_url`.
        #[arg(long)]
        lm_base_url: Option<String>,
        /// Override `reflector.lm.model`.
        #[arg(long)]
        lm_model: Option<String>,
        /// Override evaluator concurrency.
        #[arg(long)]
        concurrency: Option<usize>,
        /// Ignore any existing checkpoint in `run_dir` and start fresh.
        #[arg(long)]
        fresh: bool,
    },
    /// Scaffold a new universal `task.json` template.
    Init {
        /// Task name.
        name: String,
        #[arg(long, value_enum, default_value_t = DataMode::Generalize)]
        mode: DataMode,
        #[arg(long, value_enum, default_value_t = EvalKind::ShellCheck)]
        eval: EvalKind,
        /// Destination file path (defaults to `<name>.task.json`).
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Inspect an existing run directory (first line is a self-contained summary).
    Status {
        run_dir: PathBuf,
        /// Print full JSON summary instead of human-readable text.
        #[arg(long)]
        json: bool,
    },
    /// Create a STOP sentinel file inside `run_dir` to gracefully halt a running loop.
    Stop {
        run_dir: PathBuf,
    },
    /// Convert a legacy `gepa_plugin/tasks/*.json` file into the universal schema.
    Convert {
        legacy_task: PathBuf,
        /// Output path for the converted universal JSON (defaults to stdout).
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Python executable used to expand `{python}` placeholders.
        #[arg(long)]
        python: Option<String>,
    },
}

fn main() {
    if let Err(e) = try_main() {
        eprintln!("gepa error: {e}");
        std::process::exit(1);
    }
}

fn try_main() -> gepa_core::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run {
            task,
            run_dir,
            budget,
            max_iterations,
            lm_base_url,
            lm_model,
            concurrency,
            fresh,
        } => {
            let loaded = load_task_from_file(&task)?;
            let resolved_run_dir = run_dir.unwrap_or_else(|| {
                let stem = loaded
                    .spec
                    .name
                    .as_deref()
                    .or_else(|| task.file_stem().and_then(|s| s.to_str()))
                    .unwrap_or("run");
                PathBuf::from("runs").join(stem)
            });
            run_task(
                loaded,
                RunOptions {
                    run_dir: resolved_run_dir,
                    budget,
                    max_iterations,
                    lm_base_url,
                    lm_model,
                    concurrency,
                    fresh,
                },
            )?;
        }
        Commands::Init {
            name,
            mode,
            eval,
            output,
        } => {
            let spec = scaffold_task(&name, mode, eval);
            let out_path = output.unwrap_or_else(|| PathBuf::from(format!("{name}.task.json")));
            let json = serde_json::to_string_pretty(&spec)?;
            write_atomic(&out_path, format!("{json}\n").as_bytes())?;
            println!("Created {}", out_path.display());
        }
        Commands::Status { run_dir, json } => {
            let text = status_text(&run_dir, json)?;
            println!("{text}");
        }
        Commands::Stop { run_dir } => {
            let stop_file = stop_run(&run_dir)?;
            println!("Created stop sentinel: {}", stop_file.display());
        }
        Commands::Convert {
            legacy_task,
            output,
            python,
        } => {
            let raw = fs::read_to_string(&legacy_task)
                .map_err(|e| format!("failed to read `{}`: {e}", legacy_task.display()))?;
            let leg: LegacyTask = serde_json::from_str(&raw)?;
            let task_dir = legacy_task
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| std::path::Path::new("."));
            let (spec, warnings) = convert_legacy(leg, task_dir, python.as_deref())?;
            for w in warnings {
                eprintln!("warning: {w}");
            }
            let rendered = format!("{}\n", serde_json::to_string_pretty(&spec)?);
            if let Some(out) = output {
                write_atomic(&out, rendered.as_bytes())?;
                eprintln!("Wrote universal task spec to {}", out.display());
            } else {
                print!("{rendered}");
            }
        }
    }
    Ok(())
}

fn scaffold_task(name: &str, mode: DataMode, eval: EvalKind) -> TaskSpec {
    let dataset = match mode {
        DataMode::Generalize => DatasetSpec {
            train: Some(PathBuf::from("data/train.jsonl")),
            val: Some(PathBuf::from("data/val.jsonl")),
            test: Some(PathBuf::from("data/test.jsonl")),
            ..Default::default()
        },
        DataMode::Multi => DatasetSpec {
            train: Some(PathBuf::from("data/dataset.jsonl")),
            ..Default::default()
        },
        DataMode::Single => DatasetSpec::default(),
    };

    let evaluator = match eval {
        EvalKind::CommandJson => EvalConfig::CommandJson(JsonCommandConfig {
            program: "python".to_string(),
            args: vec!["eval.py".to_string()],
            ..Default::default()
        }),
        EvalKind::ShellCheck => EvalConfig::ShellCheck(ShellCheckConfig {
            check: "pytest -q".to_string(),
            targets: BTreeMap::from([("prompt".to_string(), PathBuf::from("prompt.txt"))]),
            ..Default::default()
        }),
        EvalKind::Http => EvalConfig::Http(HttpEvalConfig {
            url: "http://127.0.0.1:9000/evaluate".to_string(),
            ..Default::default()
        }),
    };

    let mut engine = default_engine_config();
    if mode == DataMode::Single {
        engine.minibatch_size = 1;
    }

    TaskSpec {
        schema_version: 1,
        name: Some(name.to_string()),
        mode,
        seed: BTreeMap::from([(
            "prompt".to_string(),
            "You are a helpful assistant.".to_string(),
        )]),
        seed_files: BTreeMap::new(),
        dataset,
        evaluator,
        reflector: ReflectorSpec {
            lm: Default::default(),
            prompt: gepa_reflect::PromptConfig {
                objective: Some("Maximize task evaluation score.".to_string()),
                background: None,
                custom_template: None,
                hide_keys: if mode == DataMode::Multi {
                    vec!["expected".into(), "Expected answer".into(), "answer".into()]
                } else {
                    vec![]
                },
                ..Default::default()
            },
        },
        engine,
    }
}
