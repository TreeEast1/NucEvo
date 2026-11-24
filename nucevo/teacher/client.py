"""OpenAI-compatible external reflection teacher client with disk caching and accounting.

Credentials are loaded strictly from environment variables or a local untracked `.env`
file (never logged or written to result artifacts).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path


from nucevo.config import cache_subdir


def load_env_file(path: str | Path | None = None) -> None:
    target = Path(path or os.environ.get("NUCEVO_TEACHER_ENV", ".env"))
    if not target.exists():
        return
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip("\"'"))


@dataclass
class TeacherStats:
    calls: int = 0
    cached: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    wall_time: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage, latency: float, cached: bool = False) -> None:
        with self._lock:
            if cached:
                self.cached += 1
                return
            self.calls += 1
            self.wall_time += latency
            if usage:
                self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
                details = getattr(usage, "completion_tokens_details", None)
                if details:
                    self.reasoning_tokens += int(getattr(details, "reasoning_tokens", 0) or 0)

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "cached": self.cached,
            "errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "wall_time_s": round(self.wall_time, 2),
        }


class TeacherClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 10000,
        timeout: float = 180.0,
        max_retries: int = 4,
        cache_dir: str | Path | None = None,
        max_concurrency: int = 1,
        log_path: str | Path | None = None,
        reasoning_effort: str | None = "no_think",
        min_interval: float | None = None,
    ):
        load_env_file()
        self.base_url = (
            base_url
            or os.environ.get("NUCEVO_TEACHER_BASE_URL")
            or os.environ.get("HY4_BASE_URL")
            or ""
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("NUCEVO_TEACHER_API_KEY")
            or os.environ.get("HY4_API_KEY")
            or ""
        )
        self.model = (
            model
            or os.environ.get("NUCEVO_TEACHER_MODEL")
            or os.environ.get("HY4_MODEL")
            or "gpt-4o"
        )
        if not self.base_url or not self.api_key:
            raise RuntimeError(
                "Teacher endpoint/API key not configured. Set NUCEVO_TEACHER_BASE_URL and NUCEVO_TEACHER_API_KEY."
            )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.min_interval = (
            float(min_interval)
            if min_interval is not None
            else float(os.environ.get("NUCEVO_TEACHER_MIN_INTERVAL", "2.0"))
        )
        self._last_call = 0.0
        self.cache_dir = Path(cache_dir) if cache_dir else cache_subdir("teacher")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = TeacherStats()
        self.log_path = Path(log_path) if log_path else None
        self._sem = threading.Semaphore(max_concurrency)
        from openai import OpenAI
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=timeout, max_retries=0)

    def _key(self, messages, temperature: float, max_tokens: int, effort: str | None) -> str:
        payload = json.dumps(
            {"m": messages, "t": temperature, "n": max_tokens, "model": self.model, "r": effort},
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode()).hexdigest()

    def _throttle(self) -> None:
        dt = time.time() - self._last_call
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last_call = time.time()

    def chat(
        self,
        messages,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
        response_format_json: bool = False,
        reasoning_effort: str | None = None,
    ) -> str:
        temperature = self.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.max_tokens
        effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort
        key = self._key(messages, temperature, max_tokens, effort)
        cache_path = self.cache_dir / f"{key}.json"
        if use_cache and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                self.stats.add(None, 0.0, cached=True)
                return cached["text"]
            except Exception:
                pass

        last_err = None
        for attempt in range(self.max_retries):
            try:
                self._throttle()
                with self._sem:
                    t0 = time.time()
                    kwargs = dict(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
                    if response_format_json:
                        kwargs["response_format"] = {"type": "json_object"}
                    if effort:
                        kwargs["extra_body"] = {"reasoning_effort": effort}
                    resp = self._client.chat.completions.create(**kwargs)
                    latency = time.time() - t0
                text = resp.choices[0].message.content or ""
                self.stats.add(getattr(resp, "usage", None), latency)
                self._log(
                    {
                        "ts": time.time(),
                        "latency": round(latency, 2),
                        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                        "ok": True,
                    }
                )
                if use_cache:
                    cache_path.write_text(json.dumps({"text": text, "key": key}), encoding="utf-8")
                return text
            except Exception as err:
                last_err = err
                self.stats.errors += 1
                self._log({"ts": time.time(), "ok": False, "error": str(err)[:200]})
                code = getattr(err, "status_code", None)
                base = 45 if code == 429 else 3
                time.sleep(min(base * (2**attempt), 240))
        raise RuntimeError(f"Teacher call failed after {self.max_retries} attempts: {last_err}")

    def chat_json(self, messages, **kwargs):
        return parse_json(self.chat(messages, **kwargs))

    def map(self, fn, items, workers: int = 4):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, items))

    def _log(self, rec: dict) -> None:
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec) + "\n")


def parse_json(text: str | None):
    if text is None:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(cleaned[idx:])
                return obj
            except Exception:
                continue
    return None


class MockTeacher:
    """Offline mock teacher for unit/pipeline testing."""

    def __init__(self):
        self.stats = TeacherStats()

    def chat(self, messages, **kwargs) -> str:
        self.stats.calls += 1
        return json.dumps({"experiences": []})

    def chat_json(self, messages, **kwargs) -> dict:
        return {"experiences": []}
