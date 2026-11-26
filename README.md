# gepa-rs

`gepa-rs` 是 **GEPA（Genetic-Pareto Reflective Evolution，反射式遗传帕累托优化）** 算法的 Rust 移植实现（对应 Python 版 `gepa 0.1.4` 的核心搜索引擎与 `optimize_anything` 默认策略）。

它用于优化由一个或多个命名文本组件构成的目标产物（如 System Prompt、Agent 多阶段指令、代码片段等），结合 **执行轨迹反射变异（Reflective Mutation）**、**多维帕累托前沿维护（Pareto Frontier）** 与 **谱系三路合并（Genealogical Merge）**，在有限的评估预算内高效搜索最优文本组合。

---

## 核心特性

- **Sans-IO 无副作用内核（Functional Core, Imperative Shell）**：核心库 `gepa-core` **零网络、零文件 IO、零内置大模型依赖**（仅依赖 `serde` 与 `serde_json`）。所有外部交互（运行评估、调用大模型生成新文本、事件监听）全部收敛于单一的 `Adapter` Trait。
- **多维帕累托前沿保留专长候选**：不止看验证集平均分，支持按样本（`Instance`）、按指标（`Objective`）、混合（`Hybrid`）及笛卡尔积（`Cartesian`）维护帕累托前沿，避免局部专长型候选在早期被均值淘汰。
- **基于共同祖先的谱系合并（3-Way Merge）**：追踪所有候选的演化有向无环图（DAG），自动寻找非支配池中具有共同祖先且修改互补的两个分支进行组件级三路合并，并通过分层子采样快速验真。
- **完全确定性与原生断点续跑（Checkpoint / Resume）**：全量采用 `BTreeMap` / `BTreeSet` 消除哈希随机性，内置可序列化的 `SplitMix64` 随机数生成器。任意时刻将 `Engine` 序列化为 JSON 即为完整快照，反序列化后可无缝恢复相同的随机数流与搜索轨迹。

---

## Workspace 目录与模块状态

本项目采用 Rust Workspace（Edition 2024）组织，划分为四个职责解耦的 Crate：

```text
gepa-rs/
├── gepa-core/       # [已完成] 纯算法核心引擎：状态、帕累托前沿、反射循环、谱系合并、策略与停止条件
│   ├── src/
│   │   ├── lib.rs         # 公共导出与错误类型定义
│   │   ├── engine.rs      # Engine 主循环、Adapter 接口、反射变异与合并调度
│   │   ├── state.rs       # Candidate/State 数据结构与四类帕累托前沿更新
│   │   ├── pareto.rs      # 支配剪枝 (remove_dominated) 与按前沿持有数加权采样
│   │   ├── merge.rs       # 共同祖先搜索、组件级三路合并与分层验证集子采样
│   │   ├── strategies.rs  # 父代选择、组件轮询、Epoch 分批采样、接受门控与停止器
│   │   └── rng.rs         # 可序列化的确定性 SplitMix64 伪随机数生成器
│   └── tests/
│       └── engine.rs      # 端到端收敛、确定性快照回放、支配剪枝与合并单元测试
├── gepa-reflect/    # [已完成] Adapter::propose 实现：反馈模板渲染、代码块提取与 OpenAI 兼容客户端
│   └── src/
│       ├── lib.rs         # Reflector 入口与单测
│       ├── template.rs    # PromptConfig、有序 side_info 渲染与 hide_keys 脱敏
│       ├── extract.rs     # <think> 推理块剥离与嵌套代码块提取
│       └── client.rs      # LmClient trait 与 ureq 同步 OpenAiClient（自动绕过环回代理）
├── gepa-eval/       # [已完成] Adapter::evaluate 实现：并发评估与跨平台进程树超时清理
│   └── src/
│       ├── lib.rs         # EvalConfig、Evaluator trait 与线程池并发调度
│       ├── process.rs     # 跨平台进程树强杀（Windows taskkill /T、Unix killpg）
│       ├── command.rs     # command_json：子进程 stdin/stdout JSON 协议评估器
│       ├── workspace.rs   # shell_check：隔离临时工作区、protect 保护与 GEPA_SCORE 解析
│       └── http.rs        # http：远程/本地 HTTP JSON 评估端点
└── gepa-cli/        # [已完成] `gepa` 命令行入口与通用任务 Schema（含旧版任务兼容层）
    ├── src/
    │   ├── main.rs        # clap 子命令：run / init / status / stop / convert
    │   ├── schema.rs      # 通用 TaskSpec v1（generalize / multi / single）与数据集加载
    │   ├── legacy.rs      # gepa_plugin 旧版 tasks/*.json 自动识别与转换
    │   └── runner.rs      # NDJSON 事件流、原子 checkpoint 存盘、status 摘要与 STOP 控制
    └── tests/
        └── cli.rs         # init / convert / run / resume / status / stop 端到端集成测试
```

---

## 系统架构与单步演化流程

```mermaid
flowchart TD
    Host["宿主程序 / CLI"] <-->|"实现 Adapter Trait<br/>(evaluate / propose / on_event)"| Engine["gepa-core::Engine"]

    subgraph EngineLoop ["Engine::step 单次迭代"]
        Start["迭代开始 (iteration + 1)"] --> CheckMerge{"是否存在待执行 Merge?<br/>(due > 0 且上轮有新候选)"}
        CheckMerge -- 是 --> TryMerge["谱系三路合并 (try_merge)<br/>1. 从非支配池选共同祖先分支对<br/>2. 合并互补组件修改<br/>3. 5 个分层验证集样本预筛"]
        TryMerge -- "无法构造有效合并对 (None)" --> Reflect
        TryMerge -- "预筛通过 (after >= before)" --> FullVal["全量验证集评估 (val_eval)<br/>加入 State 并更新帕累托前沿"]
        TryMerge -- "预筛未通过" --> Reject["触发 Event::Rejected"]

        CheckMerge -- 否 --> Reflect["反射变异 (reflect)<br/>1. CandidateSelector 选父代 + EpochSampler 抽训练集 Minibatch<br/>2. Adapter::evaluate 获取父代得分与 feedback 轨迹<br/>3. ComponentSelector 选目标组件 + Adapter::propose 生成新文本<br/>4. 同一 Minibatch 评估子代得分"]
        Reflect --> Gate{"Acceptance & Selection<br/>子代 Minibatch 得分门控与去重"}
        Gate -- 通过 --> FullVal
        Gate -- 未通过 --> Reject
    end

    FullVal --> EventNew["触发 Event::NewCandidate<br/>(若启用 Merge 则标记 due + 1)"]
```

---

## 核心概念说明

### 1. 数据所有权与 `Candidate`
- **宿主持有原始数据**：`gepa-core` 不存储具体的训练/验证文本，仅通过 `DataId`（`0..train_len` 与 `0..val_len` 下标）向 `Adapter` 请求评估，保持引擎状态轻量且与业务数据结构解耦。
- **多组件候选（`Candidate`）**：定义为 `BTreeMap<String, String>`（`组件名 -> 文本内容`），既支持单 Prompt 优化，也支持多模块 Agent 系统按轮询（`RoundRobin`）或全量（`All`）方式协同演化。

### 2. `Adapter` 宿主接口
接入任意评估环境与大模型只需实现 `gepa_core::Adapter`：
- `evaluate(&mut self, candidate, split, ids, capture_traces) -> Result<EvalBatch>`：在指定的 `Train` 或 `Val` 子集下标上运行候选，返回每个样本的主得分 `scores`、可选多目标指标 `objectives` 以及用于反射的诊断信息 `feedback`（仅当 `capture_traces == true` 时需要）。
- `propose(&mut self, parent, ids, eval, components) -> Result<BTreeMap<String, String>>`：根据父代在训练 Minibatch 上的运行反馈 `eval`，重写 `components` 指定的组件文本。
- `on_event(&mut self, event)`（可选）：监听 `IterationStart`、`NewCandidate`、`Rejected`、`Skipped`、`Error` 事件。

### 3. 四种帕累托前沿模式（`FrontierType`）
| 模式 | 前沿维度 | 适用场景 |
| :--- | :--- | :--- |
| `Instance` | 每个验证集样本 `Val(id)` 独立一个前沿 | 防止擅长疑难样本的特化候选被高均值平庸候选淘汰 |
| `Objective` | 每个评估指标 `Objective(name)` 独立一个前沿 | 多目标权衡（如准确率 vs. 响应长度/延迟） |
| `Hybrid` *(默认)* | `Instance` ∪ `Objective` 并集 | 兼顾单样本特化能力与多指标平衡 |
| `Cartesian` | `Cartesian(id, metric)` 样本 × 指标乘积 | 极细粒度的样本级多指标前沿追踪 |

### 4. 停止条件（`Stop`）
为防止无限消耗预算，`Engine::new` 强制要求至少配置一项停止条件（任一满足即停止）：
- `max_metric_calls`：最大评估调用次数预算
- `max_iterations`：最大迭代轮数
- `max_candidates`：最多接受的候选总数
- `score_threshold`：验证集最优均分达到阈值即停
- `max_stale_iterations`：连续多少轮未刷新验证集最高均分即停（早停）
- `timeout_secs`：墙钟超时秒数
- `stop_file`：检测到指定路径文件存在时优雅退出

---

## 快速开始

### 构建与运行测试

```bash
cargo test --workspace
```

### CLI 用法（`gepa`）

1. **生成任务模板（`init`）**：
   ```bash
   cargo run -p gepa-cli -- init my_task --mode generalize --eval shell-check -o my_task.task.json
   ```
   - `--mode`: `generalize`（`train` + `val` [+ `test`]）、`multi`（单数据集兼作训练与验证，默认隐藏 `expected`/`Expected answer`/`answer` 防止泄露答案）、`single`（无数据集单例黑盒优化）。
   - `--eval`: `shell-check`（隔离工作区写入候选文件并执行 `check` 命令）、`command-json`（任意子进程 stdin/stdout JSON 协议）、`http`（HTTP JSON 评估端点）。

2. **运行与断点续跑（`run`）**：
   ```bash
   cargo run -p gepa-cli -- run my_task.task.json --run-dir runs/my_task --budget 150
   ```
   - 标准输出实时流式输出 NDJSON 事件（`seed_evaluated`、`iteration_start`、`new_candidate`、`rejected`、`skipped`、`completed`）。
   - 每步原子落盘 `checkpoint.json`、`best_candidate.json`、`best/<component>.txt`、`run.log`，结束时写入 `summary.json`。再次对同一 `--run-dir` 执行 `gepa run` 会自动从断点恢复（加 `--fresh` 可忽略旧断点重跑）。

3. **查看状态与优雅停止（`status` / `stop`）**：
   ```bash
   cargo run -p gepa-cli -- status runs/my_task        # 首行输出自包含单行摘要，方便折叠式 UI 展示
   cargo run -p gepa-cli -- status runs/my_task --json # 输出 JSON 摘要
   cargo run -p gepa-cli -- stop runs/my_task          # 创建 STOP 文件通知引擎安全退出
   ```

4. **兼容与迁移旧版 `gepa_plugin/tasks/*.json`（`convert`）**：
   - `gepa run <legacy.json>` 会自动识别旧版 `tasks/*.json` 格式并在内存中完成兼容转换。
   - 如需持久化迁移为新版通用 Schema：
     ```bash
     cargo run -p gepa-cli -- convert legacy_task.json -o task.json --python .venv/Scripts/python
     ```

### 通用 `task.json` Schema 示例（v1）

```json
{
  "schema_version": 1,
  "name": "my_task",
  "mode": "generalize",
  "seed": {
    "system_prompt": "You are a careful problem solver."
  },
  "dataset": {
    "train": "data/train.jsonl",
    "val": "data/val.jsonl",
    "test": "data/test.jsonl"
  },
  "evaluator": {
    "type": "command_json",
    "program": "python",
    "args": ["eval_example.py"],
    "timeout_secs": 300,
    "concurrency": 4
  },
  "reflector": {
    "lm": {
      "base_url": "http://127.0.0.1:8080/v1",
      "model": "default",
      "temperature": 0.6,
      "timeout_secs": 300
    },
    "objective": "Maximize solve accuracy while keeping responses concise.",
    "background": "Optional domain guidance for the reflection LM."
  },
  "engine": {
    "minibatch_size": 3,
    "frontier": "hybrid",
    "stop": {
      "max_metric_calls": 150
    }
  }
}
```

### 在 Rust 代码中嵌入 `gepa-core`

```rust
use std::collections::BTreeMap;
use gepa_core::strategies::Stop;
use gepa_core::{Adapter, Candidate, Config, DataId, Engine, EvalBatch, Event, Split};
use serde_json::json;

struct MyAdapter;

impl Adapter for MyAdapter {
    fn evaluate(
        &mut self,
        candidate: &Candidate,
        _split: Split,
        ids: &[DataId],
        capture_traces: bool,
    ) -> gepa_core::Result<EvalBatch> {
        let prompt = &candidate["system_prompt"];
        let scores: Vec<f64> = ids
            .iter()
            .map(|&id| if prompt.contains(&id.to_string()) { 1.0 } else { 0.0 })
            .collect();
        let feedback = if capture_traces {
            ids.iter().map(|&id| json!({ "missing_id": id })).collect()
        } else {
            vec![]
        };
        Ok(EvalBatch { scores, feedback, ..Default::default() })
    }

    fn propose(
        &mut self,
        parent: &Candidate,
        _ids: &[DataId],
        eval: &EvalBatch,
        components: &[String],
    ) -> gepa_core::Result<BTreeMap<String, String>> {
        let comp = &components[0];
        let mut next_text = parent[comp].clone();
        if let Some((fb, _)) = eval.feedback.iter().zip(&eval.scores).find(|(_, s)| **s < 1.0) {
            next_text.push_str(&format!(" {}", fb["missing_id"]));
        }
        Ok(BTreeMap::from([(comp.clone(), next_text)]))
    }

    fn on_event(&mut self, event: &Event) {
        println!("{event:?}");
    }
}

fn main() -> gepa_core::Result<()> {
    let mut adapter = MyAdapter;
    let seed = Candidate::from([("system_prompt".into(), "init:".into())]);
    let config = Config {
        stop: Stop {
            max_metric_calls: Some(200),
            score_threshold: Some(1.0),
            ..Default::default()
        },
        ..Default::default()
    };

    // 初始化引擎（会对 seed 执行一次全量验证集评估）
    let mut engine = Engine::new(config, seed, /*train_len=*/ 4, /*val_len=*/ 4, &mut adapter)?;

    // 逐步执行并在每步后支持 JSON 序列化存档
    while !engine.should_stop() {
        engine.step(&mut adapter)?;
        let _checkpoint_json = serde_json::to_string(&engine)?;
    }

    let (best_idx, best_candidate) = engine.best();
    println!("Best #{best_idx}: {best_candidate:?}");
    Ok(())
}
```

---

## License

MIT
