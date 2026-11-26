---
name: gepa
description: >-
  Use this skill when the user asks to optimize, evolve, or tune an LLM/Agent system prompt,
  source code file, or GPU kernel using GEPA (gepa-rs) with execution feedback and Pareto search.
---

# GEPA Optimizer Skill (`gepa-rs`)

当用户要求用 GEPA 自动优化提示词、代码或 GPU Kernel 时，**严格按以下流程执行**。

## 硬性约束

1. **绝不凭空猜测模型接口与 Key**：如果用户未明确给出反思 LLM 的 `base_url` + `model`（以及 `command_json` 被测模型接口），先向用户确认，不要编造。
2. **禁止同目录并发写**：同一个 `--run-dir` 同时只能有一个 `gepa run` 进程。
3. **后台运行非零预算任务**：`--budget > 0` 的优化必须放后台执行，不要阻塞对话；通过 `gepa status <run_dir>`（首行即为自包含摘要）查进度，通过 `gepa stop <run_dir>` 优雅停止。

---

## Step 0：定位 `gepa` 可执行文件

按以下顺序查找 `gepa`（Windows 为 `gepa.exe`），找到即用：
1. 本 Skill 所在插件内置二进制：`<this_skill_dir>/../../bin/gepa`（或 `gepa.exe`）；
2. 系统 `PATH` 中的 `gepa`；
3. 当前工作区内的 `./target/release/gepa`（若在 `gepa-rs` 源码仓库且尚未编译，先执行 `cargo build --release -p gepa-cli`）。

---

## Step 1：从用户需求确定两项配置

1. **评估类型（`evaluator.type`）**：
   - 优化 **LLM / Agent 提示词** → 选 **`command_json`**（编写轻量 `eval.py` 桥接脚本）。
   - 直接优化 **源码 / Kernel / 配置文件** → 选 **`shell_check`**（写入隔离目录直接跑检查命令）。
2. **数据模式（`mode`）**：
   - **`generalize`** *(默认)*：需泛化到未见过的新题，提供 `train` + `val`（或只提供 `pool` 自动切分，建议总样本 ≥ 6）。
   - **`multi`**：只在这批固定样本上搜索并留优，`train` 共用为验证集（引擎默认屏蔽 `expected` / `Expected answer` / `answer` 防抄答案）。
   - **`single`**：单一问题调优（无需题库），设 `"minibatch_size": 1`；若开启 `"merge"`，需设 `"val_overlap_floor": 1`。

---

## Step 2：编写 `eval.py`（仅 `command_json` 需要）与 `task.json`

`command_json` 脚本协议（UTF-8）：
- 从 `stdin` 读取 `{"candidate": {"system_prompt": "..."}, "split": "train"|"val", "id": 0, "example": {...}, "capture_traces": true}`。
- 调目标 LLM/Agent 时（尤其内网地址）用 `urllib.request.build_opener(urllib.request.ProxyHandler({}))` 绕过代理。
- 向 `stdout` 打印单行 JSON：`{"score": <float>, "objectives": {"pass": 0|1, ...}, "side_info": {"Problem": ..., "Detail": ..., "Reply (tail)": ...}}`（`capture_traces=false` 时可省略 `side_info`）。

`shell_check` 任务注意：
- 将测试文件列入 `protect` 数组，防止候选覆盖或篡改测试作弊。
- 检查脚本在 `stdout` 打印 `GEPA_SCORE=<float>`（可选 `GEPA_SCORES={"metric": float}`）。

`task.json` 必须写清 `reflector.objective`、`reflector.background`、`reflector.lm`（`base_url`、`model`）和 `engine.stop`（如 `max_metric_calls`）。可用 `<gepa> init <name> --mode <mode> --eval <command-json|shell-check> -o task.json` 生成骨架。

---

## Step 3：两步上线前预检（必须通过才跑正式预算）

1. **单样本冒烟**：用 `train` 第 1 条数据喂给 `eval.py`（或跑一次 `check`），确认退出码为 `0` 且能解析出合法分数。
2. **零预算基线**：
   ```bash
   <gepa> run task.json --run-dir runs/<name> --budget 0 --fresh
   ```
   - 若基线已接近 `1.0`：告知用户题目过易、缺乏优化信号，建议换更难样本。
   - 若基线为 `0.0`：对照单样本输出确认是候选本身能力不足，而非脚本路径、代理或权限报错。

---

## Step 4：后台启动优化并交付结果

1. 后台运行正式预算（不加 `--fresh` 可接着刚跑完的基线断点续跑）：
   ```bash
   <gepa> run task.json --run-dir runs/<name> --budget <N>
   ```
2. 结束后读取 `runs/<name>/summary.json`、`runs/<name>/best_candidate.json` 与 `runs/<name>/best/*.txt`：
   - 向用户对比 **初始分数 vs 最优分数**（含 `test_score`、`best_objectives`、总评估数）；
   - 总结最优候选相对 `seed` 新增的关键规则或代码改动；
   - 经用户确认后将 `best/*.txt` 写入目标项目文件。
