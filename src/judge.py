from __future__ import annotations

import json
import logging
import re
from typing import Callable

from .runner import _estimate_tokens, run_model

log = logging.getLogger(__name__)

_JUDGE_SYS = (
    "You are an impartial AI judge. Evaluate model responses objectively. "
    "Return ONLY valid JSON, no other text."
)
_CHARS_PER_TOKEN = 2.0


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _per_model_chars(params: dict, n_models: int) -> int:
    ctx = params.get("ctx_size", 4096)
    max_tok = params.get("max_tokens", 512)
    sys_tok = _estimate_tokens(_JUDGE_SYS)
    overhead = 200 + 60 * n_models
    budget = max(200, ctx - sys_tok - max_tok - overhead)
    return int(budget / n_models * _CHARS_PER_TOKEN)


def run_judge_all(
    llama_path: str,
    judge_model_path: str,
    system_prompt: str,
    user_input: str,
    results: list[dict],
    params: dict,
    extra_args: str,
    proc_started: Callable | None = None,
) -> dict:
    """Feed all outputs to the judge at once.

    Returns dict with keys: mode, winner, scores (name→1-10), reasoning, raw.
    On failure: error, raw.
    """
    n = len(results)
    per_chars = _per_model_chars(params, n)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    blocks: list[str] = []
    for i, r in enumerate(results):
        letter = letters[i] if i < 26 else str(i + 1)
        out = _truncate(r.get("output", ""), per_chars)
        blocks.append(f"Model {letter} ({r['name']}):\n{out}")

    score_template = ", ".join(f'"{r["name"]}": <1-10>' for r in results)
    user_prompt = (
        f"Original prompt: {user_input}\n"
        f"System context: {system_prompt or '(none)'}\n\n"
        + "\n\n".join(blocks)
        + f'\n\nReturn ONLY this JSON:\n'
        f'{{"winner": "<model name>", "scores": {{{score_template}}}, "reasoning": "<explanation>"}}'
    )

    res = run_model(
        llama_path=llama_path,
        model_path=judge_model_path,
        system_prompt=_JUDGE_SYS,
        user_input=user_prompt,
        params=params,
        chat_mode=True,
        extra_args=extra_args,
        proc_started=proc_started,
    )

    if not res.get("success"):
        return {"error": res.get("error", "Judge failed"), "raw": res.get("output", "")}

    parsed = _extract_json(res["output"])
    if parsed is None:
        return {"error": "Could not parse JSON from judge output", "raw": res["output"]}

    return {
        "mode": "all",
        "winner": parsed.get("winner"),
        "scores": parsed.get("scores", {}),
        "reasoning": parsed.get("reasoning", ""),
        "raw": res["output"],
    }


def run_judge_pairwise(
    llama_path: str,
    judge_model_path: str,
    system_prompt: str,
    user_input: str,
    results: list[dict],
    params: dict,
    extra_args: str,
    on_match_done: Callable[[int, int, str, str, str | None, str], None] | None = None,
    stop_flag: Callable[[], bool] | None = None,
    proc_started: Callable | None = None,
) -> dict:
    """Round-robin pairwise tournament.

    on_match_done(match_idx, total, name_a, name_b, winner_name | None, reasoning)
    Returns dict with keys: mode, winner, wins (name→int), scores (name→0-100), matches.
    """
    n = len(results)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    total = len(pairs)
    per_chars = _per_model_chars(params, 2)

    wins: dict[str, int] = {r["name"]: 0 for r in results}
    matches: list[dict] = []

    for idx, (i, j) in enumerate(pairs):
        if stop_flag and stop_flag():
            break

        ra, rb = results[i], results[j]
        out_a = _truncate(ra.get("output", ""), per_chars)
        out_b = _truncate(rb.get("output", ""), per_chars)

        user_prompt = (
            f"Original prompt: {user_input}\n\n"
            f"Model A ({ra['name']}):\n{out_a}\n\n"
            f"Model B ({rb['name']}):\n{out_b}\n\n"
            f'Which is better? Return ONLY: {{"winner": "A" or "B", "reasoning": "<one sentence>"}}'
        )

        res = run_model(
            llama_path=llama_path,
            model_path=judge_model_path,
            system_prompt=_JUDGE_SYS,
            user_input=user_prompt,
            params=params,
            chat_mode=True,
            extra_args=extra_args,
            proc_started=proc_started,
        )

        winner_name: str | None = None
        reasoning = ""
        if res.get("success"):
            parsed = _extract_json(res["output"])
            if parsed:
                w = str(parsed.get("winner", "")).upper().strip().strip("\"' ")
                reasoning = parsed.get("reasoning", "")
                if w == "A":
                    winner_name = ra["name"]
                    wins[ra["name"]] += 1
                elif w == "B":
                    winner_name = rb["name"]
                    wins[rb["name"]] += 1

        matches.append({"a": ra["name"], "b": rb["name"],
                        "winner": winner_name, "reasoning": reasoning})

        if on_match_done:
            on_match_done(idx, total, ra["name"], rb["name"], winner_name, reasoning)

    max_wins = max(1, n - 1)
    scores = {name: int(wins[name] / max_wins * 100) for name in wins}
    winner = max(wins, key=wins.get) if wins else None

    return {
        "mode": "pairwise",
        "winner": winner,
        "wins": wins,
        "scores": scores,
        "matches": matches,
    }
