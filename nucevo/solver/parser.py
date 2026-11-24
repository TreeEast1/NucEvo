"""Robust tool-call parser supporting nested args and flat top-level JSON fields."""
from __future__ import annotations

import json
import re

TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
JSON_DECODER = json.JSONDecoder()

FLAT_FIELDS = {
    "read_file": ("path",),
    "edit_file": ("path", "old", "new"),
    "write_file": ("path", "content"),
    "add_connect": ("pairs",),
    "replace_connect": ("old", "new"),
    "remove_connect": ("pair",),
    "set_parameter": ("instance", "parameter", "value"),
    "run_simulation": ("stop_time",),
    "check_physics": ("stop_time",),
    "compare_to_targets": ("stop_time",),
    "diagnose": ("label",),
}


def _extract_named_call(obj: dict) -> tuple[str | None, dict]:
    name = obj["name"]
    if "args" in obj or "arguments" in obj:
        return name, obj.get("args", obj.get("arguments")) or {}
    if isinstance(name, str):
        return name, {field: obj[field] for field in FLAT_FIELDS.get(name, ()) if field in obj}
    return name, {}


def _decode(text: str):
    try:
        return JSON_DECODER.raw_decode(text)[0]
    except ValueError:
        return None


def parse_tool_call(text: str | None) -> tuple[str | None, dict]:
    """Parse one tool call from `<tool_call>...</tool_call>`, fenced JSON, or raw JSON."""
    text = text or ""
    envelope = TOOL_RE.search(text)
    candidates = [envelope.group(1)] if envelope else []
    candidates.extend(re.findall(r"```(?:json|tool_call)?\s*(\{.*?\})\s*```", text, re.S))
    for candidate in candidates:
        obj = _decode(candidate)
        if isinstance(obj, dict) and "name" in obj:
            return _extract_named_call(obj)

    legacy = re.match(r"^\s*([A-Za-z_][\w]*)\s*(\{.*\})\s*$", text, re.S)
    if legacy:
        name, body = legacy.groups()
        obj = _decode(body)
        if isinstance(obj, dict):
            if "args" in obj or "arguments" in obj:
                return name, obj.get("args", obj.get("arguments")) or {}
            return name, {key: value for key, value in obj.items() if key != "name"}

    for idx, char in enumerate(text):
        if char == "{":
            obj = _decode(text[idx:])
            if isinstance(obj, dict) and "name" in obj:
                return _extract_named_call(obj)
    return None, {}
