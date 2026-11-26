//! Response post-processing: strip reasoning traces and extract fenced code blocks.

/// Remove `<think>...</think>` blocks emitted by reasoning models (e.g. Qwen3 / DeepSeek-R1).
///
/// Unclosed `<think>` tags (without a matching `</think>`) are left untouched so content is
/// never discarded accidentally.
pub fn strip_thinking(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        let after_open = &rest[start + "<think>".len()..];
        if let Some(end) = after_open.find("</think>") {
            out.push_str(&rest[..start]);
            rest = &after_open[end + "</think>".len()..];
        } else {
            out.push_str(rest);
            return out;
        }
    }
    out.push_str(rest);
    out
}

#[derive(Clone, Copy, Debug)]
struct FenceLine {
    ticks: usize,
    has_info: bool,
    line_start: usize,
    content_start: usize,
}

fn parse_fence_line(line: &str, line_start: usize, next_line_start: usize) -> Option<FenceLine> {
    let trimmed = line.trim_start();
    let ticks = trimmed.bytes().take_while(|&b| b == b'`').count();
    if ticks < 3 {
        return None;
    }
    let after = trimmed[ticks..].trim();
    // An opening fence may carry a language/info string (without backticks);
    // a closing fence must have no trailing info text.
    if after.contains('`') {
        return None;
    }
    Some(FenceLine {
        ticks,
        has_info: !after.is_empty(),
        line_start,
        content_start: next_line_start,
    })
}

fn collect_fences(text: &str) -> Vec<FenceLine> {
    let mut fences = Vec::new();
    let mut offset = 0;
    for raw_line in text.split_inclusive('\n') {
        let line = raw_line.strip_suffix('\n').unwrap_or(raw_line).strip_suffix('\r').unwrap_or(raw_line);
        let next_offset = offset + raw_line.len();
        if let Some(f) = parse_fence_line(line, offset, next_offset) {
            fences.push(f);
        }
        offset = next_offset;
    }
    if offset < text.len() {
        let line = &text[offset..];
        if let Some(f) = parse_fence_line(line, offset, text.len()) {
            fences.push(f);
        }
    }
    fences
}

/// Extract the proposed component text from an LM response.
///
/// Strips `<think>...</think>` blocks first. If an outer fence of 4+ backticks exists with a
/// matching closing fence of the same backtick count, returns its interior. Otherwise, if
/// triple-backtick fences exist, extracts between the first opening fence line and the last
/// closing fence line (preserving inner triple-backtick examples). Falls back to trimming the
/// whole post-thinking response.
pub fn extract_code_block(response: &str) -> String {
    let cleaned = strip_thinking(response);
    let fences = collect_fences(&cleaned);

    // Prefer an outer fence with >= 4 backticks if a matching closing fence exists.
    if let Some(open_idx) = fences.iter().position(|f| f.ticks >= 4) {
        let open = fences[open_idx];
        if let Some(close) = fences[open_idx + 1..]
            .iter()
            .rev()
            .find(|f| !f.has_info && f.ticks == open.ticks)
            && open.content_start <= close.line_start
        {
            return trim_trailing_newline(&cleaned[open.content_start..close.line_start]).to_string();
        }
    }

    // Fallback to outer triple-backtick pair (first opening fence -> last closing fence).
    if fences.len() >= 2 {
        let open = fences[0];
        if let Some(close) = fences[1..].iter().rev().find(|f| !f.has_info && f.ticks >= open.ticks)
            && open.content_start <= close.line_start
        {
            return trim_trailing_newline(&cleaned[open.content_start..close.line_start]).to_string();
        }
    }

    cleaned.trim().to_string()
}

fn trim_trailing_newline(s: &str) -> &str {
    let s = s.strip_suffix('\n').unwrap_or(s);
    s.strip_suffix('\r').unwrap_or(s)
}
