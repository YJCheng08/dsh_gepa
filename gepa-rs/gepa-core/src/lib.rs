//! GEPA core: reflective evolution over a Pareto frontier, with no IO and no LM.
//!
//! Port of gepa 0.1.4 (`gepa.core.{engine,state}`, `gepa.strategies`, `gepa.proposer.merge`,
//! `gepa.utils.stop_condition`). The host supplies evaluation and text proposal through
//! [`Adapter`]; everything else (parent selection, minibatching, acceptance, merge, stop)
//! lives here. [`Engine`] is serde-serializable, so a checkpoint is just its JSON.

pub mod engine;
pub mod merge;
pub mod pareto;
pub mod rng;
pub mod state;
pub mod strategies;

pub use engine::{Adapter, Config, Engine, EvalBatch, Event, Split};
pub use state::{Candidate, DataId, FrontierType, State};

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;
