# Agent.md — 编码 Agent 使用 `gepa-rs` 的标准操作指南

本文写给编码 Agent（Claude Code、Cursor、DSH、Antigravity 等）。当用户给你一个 `gepa` 可执行文件（或当前仓库），并用一句话提出“用 GEPA 优化我的提示词 / 代码 / Kernel”时，**严格按以下 4 步执行**。

---

## 硬性约束（必读）

1. **绝不凭空猜测模型接口与 Key**：如果用户没有明确给出反思 LLM 的 `base_url` + `model`（以及 `command_json` 被测模型接口），必须先询问用户，不要自己编造。
2. **禁止同目录并发写**：同一个 `--run-dir` 同时只能运行一个 `gepa run` 进程，否则 `checkpoint.json` 会互相覆盖。
3. **后台运行非零预算任务**：`--budget > 0` 的任务通常耗时数分钟到数小时，必须放到后台执行，不要阻塞对话；用 `gepa status <run_dir>`（首行即为自包含摘要）观察进度，用 `gepa stop <run_dir>` 优雅停止。

---

## Step 1：从用户的一句话中确认两项决策

1. **决定评估类型（`evaluator.type`）**：
   - 用户要优化 **LLM / Agent 的提示词** → 选 **`command_json`**（你负责写一个轻量 `eval.py` 桥接脚本）。
   - 用户要优化 **一份源码 / Kernel / 配置文件** → 选 **`shell_check`**（把文件写入隔离工作区直接跑检查命令）。
2. **决定数据模式（`mode`）**：
   - **`generalize`** *(默认)*：产出的 Prompt/代码需要泛化到未见过的新题。需提供 `train` + `val`（或只提供单一 `pool` 由引擎自动切分，建议总样本 ≥ 6）。
   - **`multi`**：只在这批固定样本上搜索并留优。`train` 共用为验证集，引擎默认屏蔽 `expected` / `Expected answer` / `answer` 防止反思模型偷抄答案。
   - **`single`**：只针对单个固定问题调优（无需数据集文件）。必须设置 `"minibatch_size": 1`；若开启 `"merge"`，需设 `"val_overlap_floor": 1`。

---

## Step 2：编写桥接脚本（仅 `command_json` 需要）与 `task.json`

### 2.1 `eval.py` 最小标准模板（仅依赖 Python 标准库）

当用户提供“数据 + 被测 LLM/Agent + 判分规则”时，按以下模板生成 `eval.py`（注意内网地址必须用 `ProxyHandler({})` 绕过系统代理，并在 `side_info` 返回精简可读的诊断信息）：

```python
import json, sys, urllib.request

SOLVER_URL = "http://<host>:<port>/v1/chat/completions"
SOLVER_MODEL = "<model-id>"

def call_solver(system_prompt: str, user_prompt: str) -> str:
    body = json.dumps({
        "model": SOLVER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.6,
    }).encode("utf-8")
    req = urllib.request.Request(SOLVER_URL, data=body, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 绕过内网代理
    with opener.open(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"] or ""

def main():
    req = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    prompt = req["candidate"]["system_prompt"]
    ex = req["example"]
    capture_traces = bool(req.get("capture_traces", True))

    reply = call_solver(prompt, ex["problem"])
    # TODO: 替换为用户的实际判分逻辑，返回 (score: float, objectives: dict, detail: str)
    score, passed, detail = 0.0, False, "replace with real grader"

    out = {"score": float(score), "objectives": {"pass": 1.0 if passed else 0.0}}
    if capture_traces:
        out["side_info"] = {
            "Problem": ex["problem"][:1200],
            "Detail": detail[-1500:],
            "Reply (tail)": reply[-1500:],
        }
    print(json.dumps(out, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
```

### 2.2 生成 `task.json`

可用 `gepa init <name> --mode <mode> --eval <command-json|shell-check> -o task.json` 生成骨架后填充：
- **`reflector.objective`**：用一句话写清优化目标（反思模型每轮都会看到）。
- **`reflector.background`**：写明打分规则、输出格式契约与已知高频报错原因，越具体越好。
- **`protect`**（仅 `shell_check`）：如果测试脚本位于 `template_dir` 内，必须列入 `protect` 数组防止被候选覆盖或篡改。
- **`engine.stop`**：至少设置 `max_metric_calls`（推荐冒烟 10–20、正式 50–150）。

---

## Step 3：两步上线前预检（必须通过才跑正式预算）

1. **单样本冒烟测试**：
   - `command_json`：取 `train` 第一行 JSON 手动构造一次 `stdin` 喂给 `python -X utf8 eval.py`，确认退出码为 `0` 且 `stdout` 能解析出合法的 `{"score": ...}`。
   - `shell_check`：确认初始文件能正常执行 `check` 并输出 `GEPA_SCORE=` 或符合预期的退出码。
2. **零预算基线验证**：
   ```bash
   gepa run task.json --run-dir runs/<name> --budget 0 --fresh
   ```
   检查输出的 `seed_evaluated` 与 `runs/<name>/summary.json`：
   - **若基线已经接近 `1.0`（满分）**：说明题目太简单、缺少优化信号，告知用户换用更难的数据集或更严苛的评分指标。
   - **若基线为 `0.0`**：对照第 1 步单样本输出，确认 `0.0` 是候选本身能力不足（正是 GEPA 要改进的信号），而非脚本路径错误、代理拦截或权限问题。

---

## Step 4：后台启动优化并交付结果

1. **启动优化**（放到后台执行，记录任务 ID）：
   ```bash
   gepa run task.json --run-dir runs/<name> --budget <N>
   ```
   *(若上一步刚跑完 `--budget 0`，直接不加 `--fresh` 即可接着断点续跑；或加 `--fresh` 重跑。)*
2. **查看与控制**：
   - 查看状态：`gepa status runs/<name>`（支持 `--json`）
   - 中途停止：`gepa stop runs/<name>`
3. **汇报与落地**：
   优化完成后，读取 `runs/<name>/summary.json`、`runs/<name>/best_candidate.json` 与 `runs/<name>/best/*.txt`：
   - 向用户汇报：**初始分数 vs 最优分数**（含 `test_score` 和各子指标 `best_objectives`）、总评估次数与候选数量；
   - 对比 `seed` 与 `best/*.txt` 总结 GEPA 提炼出的核心改动点；
   - 经用户确认后，将 `runs/<name>/best/<component>.txt` 应用到用户实际项目文件或目标 Prompt 中。
