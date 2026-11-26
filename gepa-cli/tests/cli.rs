use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use serde_json::{Value, json};

fn gepa_bin() -> &'static str {
    env!("CARGO_BIN_EXE_gepa")
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("gepa_cli_test_{name}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn init_and_convert_legacy_task_work() {
    let dir = temp_dir("init_convert");
    let init_path = dir.join("demo.task.json");
    let st = Command::new(gepa_bin())
        .args(["init", "demo", "--mode", "multi", "--eval", "command-json", "-o"])
        .arg(&init_path)
        .status()
        .unwrap();
    assert!(st.success());
    let init_json: Value = serde_json::from_str(&fs::read_to_string(&init_path).unwrap()).unwrap();
    assert_eq!(init_json["schema_version"], 1);
    assert_eq!(init_json["mode"], "multi");
    assert_eq!(init_json["evaluator"]["type"], "command_json");

    // Convert a legacy gepa_plugin task.json
    let legacy_path = dir.join("legacy.json");
    let converted_path = dir.join("converted.json");
    fs::write(
        &legacy_path,
        serde_json::to_string_pretty(&json!({
            "mode": "single",
            "candidate": "file",
            "target": "solution.txt",
            "score": "command",
            "check": "{python} -c \"print('GEPA_SCORE=1.0')\"",
            "protect": ["tests/hidden.py"],
            "objective": "Solve puzzle",
            "seed": "init",
            "budget": 20
        }))
        .unwrap(),
    )
    .unwrap();

    let st2 = Command::new(gepa_bin())
        .args(["convert"])
        .arg(&legacy_path)
        .args(["-o"])
        .arg(&converted_path)
        .args(["--python", "py3"])
        .status()
        .unwrap();
    assert!(st2.success());
    let conv: Value = serde_json::from_str(&fs::read_to_string(&converted_path).unwrap()).unwrap();
    assert_eq!(conv["mode"], "single");
    assert_eq!(conv["evaluator"]["type"], "shell_check");
    assert_eq!(conv["evaluator"]["targets"]["target"], "solution.txt");
    assert!(conv["evaluator"]["check"].as_str().unwrap().starts_with("py3 "));
    assert_eq!(conv["engine"]["stop"]["max_metric_calls"], 20);

    // Also test converting a kernelbench.json-style generalize task (pool + array test + stop_at_score).
    let kb_legacy_path = dir.join("kb_legacy.json");
    let kb_conv_path = dir.join("kb_converted.json");
    fs::write(
        &kb_legacy_path,
        serde_json::to_string_pretty(&json!({
            "mode": "generalize",
            "candidate": "agents_md",
            "objective": "Optimize CUDA kernel prompt",
            "seed": "You are a CUDA engineer.",
            "score": "command",
            "check": "\"{python}\" \"{task_dir}/check.py\"",
            "data": "data.jsonl",
            "test": ["test.jsonl"],
            "stop_at_score": 1.0,
            "check_timeout": 900
        }))
        .unwrap(),
    )
    .unwrap();
    let st3 = Command::new(gepa_bin())
        .args(["convert"])
        .arg(&kb_legacy_path)
        .args(["-o"])
        .arg(&kb_conv_path)
        .status()
        .unwrap();
    assert!(st3.success());
    let kb_conv: Value = serde_json::from_str(&fs::read_to_string(&kb_conv_path).unwrap()).unwrap();
    assert_eq!(kb_conv["mode"], "generalize");
    assert_eq!(kb_conv["dataset"]["pool"], "data.jsonl");
    assert_eq!(kb_conv["dataset"]["test"], "test.jsonl");
    assert_eq!(kb_conv["engine"]["stop"]["score_threshold"], 1.0);
    assert_eq!(kb_conv["evaluator"]["type"], "command_json");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn end_to_end_run_resume_status_and_stop() {
    let dir = temp_dir("e2e");
    let run_dir = dir.join("run_out");
    let task_path = dir.join("task.json");

    // Mock OpenAI server: first call appends "good", subsequent calls append "x".
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_srv = Arc::clone(&calls);

    let srv = thread::spawn(move || {
        // Serve up to 8 requests; stop once listener drops or enough calls finish.
        for stream in listener.incoming().take(6) {
            let Ok(mut stream) = stream else { break };
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let n = calls_srv.fetch_add(1, Ordering::SeqCst);
            let proposed = if n == 0 { "good" } else { "good_extra" };
            let body = json!({
                "choices": [{
                    "message": {
                        "content": format!("```\n{proposed}\n```")
                    }
                }]
            })
            .to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });

    // Shell check scores 1.0 iff solution.txt contains "good", 0.0 otherwise.
    #[cfg(windows)]
    let check_cmd = "findstr /C:\"good\" solution.txt >nul && echo GEPA_SCORE=1.0 || echo GEPA_SCORE=0.0";
    #[cfg(unix)]
    let check_cmd = "grep -q good solution.txt && echo GEPA_SCORE=1.0 || echo GEPA_SCORE=0.0";

    // Legacy single-mode task auto-detected by `gepa run`.
    let legacy_task = json!({
        "name": "toy",
        "mode": "single",
        "candidate": "file",
        "target": "solution.txt",
        "score": "command",
        "check": check_cmd,
        "objective": "Include keyword good",
        "seed": "bad",
        "budget": 0
    });
    fs::write(&task_path, serde_json::to_string_pretty(&legacy_task).unwrap()).unwrap();

    // 1) Budget 0: seed evaluation only.
    let out0 = Command::new(gepa_bin())
        .args(["run"])
        .arg(&task_path)
        .args(["--run-dir"])
        .arg(&run_dir)
        .args(["--budget", "0"])
        .output()
        .unwrap();
    assert!(out0.status.success(), "stderr: {}", String::from_utf8_lossy(&out0.stderr));
    let stdout0 = String::from_utf8_lossy(&out0.stdout);
    assert!(stdout0.contains("\"seed_evaluated\""));
    assert!(stdout0.contains("\"completed\""));
    assert!(run_dir.join("checkpoint.json").exists());
    assert!(run_dir.join("summary.json").exists());

    // 2) Resume with higher budget and mock LM server: finds "good" with val_score=1.0.
    let out1 = Command::new(gepa_bin())
        .args(["run"])
        .arg(&task_path)
        .args(["--run-dir"])
        .arg(&run_dir)
        .args(["--budget", "10", "--max-iterations", "2", "--lm-base-url"])
        .arg(format!("http://{addr}/v1"))
        .output()
        .unwrap();
    assert!(out1.status.success(), "stderr: {}", String::from_utf8_lossy(&out1.stderr));
    let stdout1 = String::from_utf8_lossy(&out1.stdout);
    assert!(stdout1.contains("\"resumed\""), "stdout:\n{stdout1}");
    assert!(stdout1.contains("\"new_candidate\""), "stdout:\n{stdout1}");

    let best_txt = fs::read_to_string(run_dir.join("best").join("target.txt")).unwrap();
    assert_eq!(best_txt, "good");

    // 3) Check `gepa status` first-line summary.
    let st_out = Command::new(gepa_bin())
        .args(["status"])
        .arg(&run_dir)
        .output()
        .unwrap();
    assert!(st_out.status.success());
    let status_s = String::from_utf8_lossy(&st_out.stdout);
    let line1 = status_s.lines().next().unwrap();
    assert!(line1.starts_with("gepa status: "), "line1={line1}");
    assert!(line1.contains("val=1.0000"), "line1={line1}");

    // 4) Check `gepa stop` creates STOP sentinel.
    let stop_out = Command::new(gepa_bin())
        .args(["stop"])
        .arg(&run_dir)
        .output()
        .unwrap();
    assert!(stop_out.status.success());
    assert!(run_dir.join("STOP").exists());

    drop(srv);
    let _ = fs::remove_dir_all(&dir);
}
