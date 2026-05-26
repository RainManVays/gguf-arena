import logging
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

# 2 chars per token — conservative estimate covering Cyrillic-heavy text.
_CHARS_PER_TOKEN = 2.0


def _estimate_tokens(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN + 0.5)


def truncate_user_input(system_prompt: str, user_input: str,
                        ctx_size: int, max_tokens: int) -> tuple[str, bool]:
    """Return (truncated_user_input, was_truncated).

    Trims user_input so that sys_tokens + user_tokens + max_tokens <= ctx_size.
    Raises ValueError if system prompt + max_tokens already fills the context,
    or if user_input trims down to nothing.
    """
    sys_tokens  = _estimate_tokens(system_prompt)
    user_tokens = _estimate_tokens(user_input)

    if sys_tokens + user_tokens + max_tokens <= ctx_size:
        return user_input, False

    user_budget = ctx_size - sys_tokens - max_tokens
    if user_budget <= 0:
        raise ValueError(
            f"System prompt (~{sys_tokens} tok) + max_tokens={max_tokens} "
            f"already fills ctx_size={ctx_size}. Shorten the system prompt."
        )

    max_chars = int(user_budget * _CHARS_PER_TOKEN)
    trimmed = user_input[:max_chars]
    if not trimmed:
        raise ValueError(
            f"User input trimmed to nothing: system prompt (~{sys_tokens} tok) "
            f"leaves no space with ctx_size={ctx_size} and max_tokens={max_tokens}."
        )

    log.warning(
        "user_input truncated: ~%d → ~%d tokens (ctx=%d, max_new=%d, sys=%d)",
        user_tokens, _estimate_tokens(trimmed), ctx_size, max_tokens, sys_tokens,
    )
    return trimmed, True


def build_cmd(llama_path: str, model_path: str, system_prompt: str,
              user_input: str, params: dict, chat_mode: bool,
              extra_args: str) -> list[str]:
    cmd = [
        llama_path,
        "-m", model_path,
        "-p", user_input or " ",   # llama-cli needs non-empty prompt
        "-n", str(params["max_tokens"]),
        "-c", str(params["ctx_size"]),
        "--temp", str(params["temperature"]),
        "--top-p", str(params["top_p"]),
        "-t", str(params["threads"]),
        "-ngl", str(params["gpu_layers"]),
        "--no-display-prompt",
    ]

    if system_prompt.strip():
        cmd += ["-sys", system_prompt]

    if chat_mode:
        cmd += ["-cnv", "-st", "--simple-io"]

    if params.get("top_k", 40) > 0:
        cmd += ["--top-k", str(params["top_k"])]
    if params.get("repeat_penalty", 1.0) != 1.0:
        cmd += ["--repeat-penalty", str(params["repeat_penalty"])]
    if params.get("seed", -1) != -1:
        cmd += ["--seed", str(params["seed"])]

    if extra_args.strip():
        cmd += shlex.split(extra_args)

    return cmd


def _strip_chat_noise(stdout: str) -> str:
    """Extract model response from llama-cli chat-mode stdout.

    Chat mode wraps the response in: logo/banner → '> prompt' → response → stats → 'Exiting...'.
    We grab only the lines between the '> ' marker and the trailing stats/exit lines.
    """
    lines = stdout.split('\n')

    # Find where response starts: first line after '> <user input>'
    start = 0
    for i, line in enumerate(lines):
        if line.startswith('> '):
            start = i + 1
            break

    # Find where response ends: stats block or exit message
    end = len(lines)
    for i in range(start, len(lines)):
        s = lines[i].strip()
        if s.startswith('[ Prompt:') or s in ('Exiting...', 'Exiting'):
            end = i
            break

    response_lines = lines[start:end]
    # Some llama.cpp builds prefix each response line with '|- '
    response_lines = [
        line[3:] if line.startswith('|- ') else line
        for line in response_lines
    ]
    return '\n'.join(response_lines).strip()


def _parse_stats(stderr: str) -> tuple[float | None, int | None]:
    """Return (tokens_per_second, eval_token_count) from llama.cpp stderr."""
    m = re.search(
        r"eval time\s*=.*?/\s*(\d+)\s*runs.*?([\d.]+)\s*tokens per second",
        stderr,
    )
    if m:
        return float(m.group(2)), int(m.group(1))
    return None, None


def run_model(
    llama_path: str,
    model_path: str,
    system_prompt: str,
    user_input: str,
    params: dict,
    chat_mode: bool,
    extra_args: str,
    proc_started: Callable[[subprocess.Popen], None] | None = None,
) -> dict:
    """Run llama-cli synchronously; call proc_started(proc) once the process is alive."""
    cmd = build_cmd(llama_path, model_path, system_prompt, user_input,
                    params, chat_mode, extra_args)
    model_name = Path(model_path).name
    log.info("START  model=%s  chat_mode=%s  params=%s", model_name, chat_mode, params)
    log.debug("CMD  %s", " ".join(cmd))

    t0 = time.monotonic()
    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # detach from controlling TTY so llama.cpp can't write to /dev/tty
        )
        if proc_started is not None:
            proc_started(proc)

        try:
            stdout, stderr = proc.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            proc.terminate()
            stdout, stderr = proc.communicate()
            elapsed = time.monotonic() - t0
            log.error("TIMEOUT  model=%s  after 600s", model_name)
            return {"success": False, "error": "Timeout (600 s)",
                    "elapsed": elapsed, "stderr": stderr}

        elapsed = time.monotonic() - t0
        rc = proc.returncode

        # Killed by signal (user stop or OS)
        if rc < 0:
            log.info("STOPPED  model=%s  signal=%d  elapsed=%.1fs", model_name, -rc, elapsed)
            return {
                "success": False,
                "error": "Stopped",
                "output": stdout.strip(),
                "elapsed": elapsed,
                "stderr": stderr,
                "returncode": rc,
            }

        tps, n_tokens = _parse_stats(stderr)
        output = _strip_chat_noise(stdout) if chat_mode else stdout.strip()

        if rc != 0:
            log.warning("DONE  model=%s  rc=%d  elapsed=%.1fs  stderr_tail=%s",
                        model_name, rc, elapsed, stderr[-400:].replace("\n", " "))
        else:
            log.info("DONE  model=%s  elapsed=%.1fs  tps=%s  tokens=%s",
                     model_name, elapsed,
                     f"{tps:.1f}" if tps else "?",
                     n_tokens or "?")

        if stderr:
            log.debug("STDERR  model=%s\n%s", model_name, stderr)

        return {
            "success": rc == 0,
            "output": output,
            "elapsed": elapsed,
            "tps": tps,
            "n_tokens": n_tokens,
            "stderr": stderr,
            "returncode": rc,
        }

    except Exception as exc:
        elapsed = time.monotonic() - t0
        log.exception("EXCEPTION  model=%s  elapsed=%.1fs", model_name, elapsed)
        if proc is not None and proc.poll() is None:
            proc.terminate()
        return {"success": False, "error": str(exc), "elapsed": elapsed}
