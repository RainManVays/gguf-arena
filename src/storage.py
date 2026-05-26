import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

STAND_DIR = Path.home() / ".llm_stand"
REGISTRY_DIR = Path(__file__).parent.parent / "proms_reg"
CONFIG_FILE = STAND_DIR / "config.json"
HISTORY_FILE = STAND_DIR / "history.json"

DEFAULT_CONFIG = {
    "llama_path": shutil.which("llama-cli") or "",
    "models_dir": "",
    "nlp_models_dir": "",
    "params": {
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 512,
        "ctx_size": 8192,
        "threads": -1,
        "gpu_layers": -1,
        "seed": -1,
        "top_k": 40,
        "repeat_penalty": 1.1,
    },
    "chat_mode": True,
    "extra_args": "",
}


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                data = json.load(f)
            cfg = _merge(DEFAULT_CONFIG, data)
            log.debug("Config loaded from %s", CONFIG_FILE)
            return cfg
        except Exception:
            log.exception("Failed to load config from %s, using defaults", CONFIG_FILE)
    log.info("No config file found, using defaults")
    return dict(DEFAULT_CONFIG)


def save_config(config: dict) -> None:
    STAND_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    log.debug("Config saved to %s", CONFIG_FILE)


def load_history() -> list:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def append_history(entry: dict) -> None:
    history = load_history()
    history.insert(0, entry)
    history = history[:100]
    STAND_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    log.debug("History saved (%d entries)", len(history))


def _slug(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", name.lower().strip()).strip("_") or "prompt"


def save_to_registry(name: str, system: str, user: str = "",
                     tags: list[str] | None = None) -> Path:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    path = REGISTRY_DIR / f"{_slug(name)}.yaml"
    data: dict = {"name": name, "system": system}
    if user:
        data["user"] = user
    if tags:
        data["tags"] = tags
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False,
                  sort_keys=False)
    log.info("Saved registry entry %s → %s", name, path)
    return path


def delete_from_registry(name: str) -> None:
    path = REGISTRY_DIR / f"{_slug(name)}.yaml"
    if path.exists():
        path.unlink()
        log.info("Deleted registry entry %s → %s", name, path)


def migrate_prompts_to_registry(cfg: dict) -> None:
    old = cfg.pop("prompts", [])
    if not old:
        return
    for p in old:
        name   = p.get("name", "")
        system = p.get("system", "")
        user   = p.get("user", "")
        if name and system:
            target = REGISTRY_DIR / f"{_slug(name)}.yaml"
            if not target.exists():
                save_to_registry(name, system, user)
    save_config(cfg)
    log.info("Migrated %d prompt(s) to registry", len(old))


def load_registry() -> list[dict]:
    entries = []
    for path in sorted(REGISTRY_DIR.glob("*.yaml")):
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "name" in data and "system" in data:
                entries.append(data)
        except Exception:
            log.warning("Failed to load registry file %s", path)
    return entries


def delete_history_entry(timestamp: str) -> None:
    history = load_history()
    history = [e for e in history if e.get("timestamp") != timestamp]
    STAND_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    log.debug("History entry deleted (ts=%s), %d entries remain", timestamp, len(history))


def make_history_entry(system_prompt: str, user_input: str, params: dict,
                       chat_mode: bool, model_results: list) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "system_prompt": system_prompt,
        "user_input": user_input,
        "params": params,
        "chat_mode": chat_mode,
        "models": model_results,
    }
