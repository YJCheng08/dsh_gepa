//! Cross-platform subprocess execution with piped UTF-8 I/O and process-tree timeout kill.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct RunOutput {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
}

/// Run `cmd` with optional `stdin_bytes` and kill the entire process tree if `timeout` elapses.
pub fn run_command_with_timeout(
    cmd: &mut Command,
    stdin_bytes: Option<&[u8]>,
    timeout: Duration,
) -> gepa_core::Result<RunOutput> {
    cmd.stdin(if stdin_bytes.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let mut child = cmd.spawn().map_err(|e| format!("failed to spawn command: {e}"))?;
    let pid = child.id();

    let stdin_pipe = child.stdin.take();
    let stdout_pipe = child.stdout.take().ok_or("missing stdout pipe")?;
    let stderr_pipe = child.stderr.take().ok_or("missing stderr pipe")?;

    let stdin_owned = stdin_bytes.map(|b| b.to_vec());
    let stdin_handle = stdin_pipe.map(|mut pipe| {
        thread::spawn(move || {
            if let Some(bytes) = stdin_owned {
                let _ = pipe.write_all(&bytes);
                let _ = pipe.flush();
            }
            drop(pipe);
        })
    });

    let out_handle = thread::spawn(move || read_all(stdout_pipe));
    let err_handle = thread::spawn(move || read_all(stderr_pipe));

    let deadline = Instant::now() + timeout;
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break st,
            Ok(None) => {
                if Instant::now() >= deadline {
                    timed_out = true;
                    kill_process_tree(pid);
                    let _ = child.kill();
                    break child.wait().map_err(|e| format!("wait after kill failed: {e}"))?;
                }
                thread::sleep(Duration::from_millis(15));
            }
            Err(e) => {
                kill_process_tree(pid);
                let _ = child.kill();
                return Err(format!("try_wait failed: {e}").into());
            }
        }
    };

    if let Some(h) = stdin_handle {
        let _ = h.join();
    }
    let stdout_bytes = out_handle.join().unwrap_or_default();
    let stderr_bytes = err_handle.join().unwrap_or_default();

    Ok(RunOutput {
        exit_code: status.code(),
        stdout: String::from_utf8_lossy(&stdout_bytes).into_owned(),
        stderr: String::from_utf8_lossy(&stderr_bytes).into_owned(),
        timed_out,
    })
}

fn read_all(mut reader: impl Read) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = reader.read_to_end(&mut buf);
    buf
}

fn kill_process_tree(pid: u32) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .creation_flags(CREATE_NO_WINDOW)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(unix)]
    {
        let _ = Command::new("kill")
            .args(["-s", "KILL", "--", &format!("-{pid}")])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}
