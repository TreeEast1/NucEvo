"""OpenAI-compatible client for the frozen solver, with accounting and retries."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path



@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    errors: int = 0

    def add(self, u):
        if u:
            self.prompt_tokens += int(getattr(u, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(u, "completion_tokens", 0) or 0)
        self.calls += 1

    def to_dict(self):
        return {"prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "calls": self.calls, "errors": self.errors}


class SolverClient:
    """Thin wrapper over an OpenAI-compatible endpoint (vLLM)."""

    def __init__(self, base_url="http://127.0.0.1:8011/v1", model="solver",
                 api_key="EMPTY", temperature=0.0, top_p=1.0, max_tokens=512,
                 seed=0, timeout=180.0, max_retries=3, stop=None,
                 log_path=None):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        self.stop = stop or []
        self.max_retries = max_retries
        self.usage = Usage()
        self.log_path = Path(log_path) if log_path else None
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout,
                             max_retries=0)

    def chat(self, messages, temperature=None, max_tokens=None, stop=None):
        last_err = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=self.temperature if temperature is None else temperature,
                    top_p=self.top_p,
                    max_tokens=max_tokens or self.max_tokens,
                    seed=self.seed,
                    stop=stop or self.stop or None,
                )
                text = (resp.choices[0].message.content or "")
                for s in (stop or self.stop or []):
                    if s and not text.rstrip().endswith(s):
                        # vLLM strips the stop string; restore the delimiter so
                        # the tool-call parser sees a complete block.
                        if s == "</tool_call>":
                            text = text + s
                self.usage.add(getattr(resp, "usage", None))
                rec = {"ts": time.time(), "latency": time.time() - t0,
                       "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                       "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                       "finish_reason": resp.choices[0].finish_reason}
                self._log(rec)
                return text
            except Exception as e:                     # noqa: BLE001
                last_err = e
                self.usage.errors += 1
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"solver call failed after {self.max_retries} attempts: {last_err}")

    def _log(self, rec):
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")


class MockClient:
    """Deterministic offline client for pipeline tests (no GPU)."""

    def __init__(self, script=None):
        self.usage = Usage()
        self.script = script or []
        self.i = 0

    def chat(self, messages, **kw):
        self.usage.add(None)
        if self.i < len(self.script):
            out = self.script[self.i]
        else:
            out = '<tool_call>{"name": "submit", "args": {}}</tool_call>'
        self.i += 1
        return out
