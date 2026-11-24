"""Single-episode agent loop over the Phase 7 tool environment."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .env import TaskEnv

MAX_PROMPT_CHARS = 48000        # ~12k tokens; the server context is 16k


def _truncate(history, keep=8):
    """Keep the system prompt and the first user turn, plus the last `keep` turns.
    Dropped turns are summarised so the model knows how many it lost."""
    if len(history) <= keep:
        return history, 0
    dropped = len(history) - keep
    return ([("<steps omitted>", f"[{dropped} earlier steps omitted to fit the context]")]
            + history[-keep:]), dropped


def max_steps_for(task):
    """Assembly of a large plant needs 1 inspect + 2 sims + a few edits; repair
    needs fewer.  The budget is set from structure, not from a global constant."""
    if task["family"] == "assemble":
        return min(30, 10 + task.get("n_connects", 5))
    if task["family"] == "multistep":
        return 22
    return 16


def run_episode(task, client, memory_text="", max_steps=None, seed_note=None,
                save_dir=None, episode_text="", capture_prompt=False):
    env = TaskEnv(task, max_steps=max_steps or max_steps_for(task))
    history = []
    t0 = time.time()
    max_steps = env.max_steps
    for step in range(max_steps):
        hist, dropped = _truncate(history)
        messages = env.messages(hist, memory_text=memory_text,
                                episode_text=episode_text)
        if dropped:
            messages[1]["content"] = messages[1]["content"].replace(
                "=== TaskModel.mo ===", "[context truncated]\n=== TaskModel.mo ===")
        prompt_chars = sum(len(m["content"]) for m in messages)
        if prompt_chars > MAX_PROMPT_CHARS:
            hist, _ = _truncate(history, keep=4)
            messages = env.messages(hist, memory_text=memory_text,
                                    episode_text=episode_text)
        try:
            reply = client.chat(messages)
        except Exception as e:                      # noqa: BLE001
            return _fail(task, env, history, f"solver_error: {e}", t0)
        obs, done, _ = env.step(reply)
        history.append((reply, obs))
        if done:
            break
    result = env.evaluate()
    result["tool_log"] = env.tool_log
    result["history"] = [{"assistant": a, "observation": o} for a, o in history]
    result["wall_time"] = time.time() - t0
    result["episode_steps"] = env.step_i
    result["task_id"] = task["task_id"]
    result["template"] = task["template"]
    result["family"] = task["family"]
    result["split"] = task["split"]
    result["seed_note"] = seed_note
    result["memory_chars"] = len(memory_text or "")
    result["episode_chars"] = len(episode_text or "")
    if capture_prompt:
        # the exact first user turn, which carries both injected blocks
        result["initial_prompt"] = env.initial_user_message(memory_text, episode_text)
    if save_dir:
        _save(save_dir, task, result, memory_text)
    return result


def _fail(task, env, history, err, t0):
    return {"task_id": task["task_id"], "template": task["template"],
            "family": task["family"], "split": task["split"], "success": False,
            "error": err, "tool_log": env.tool_log,
            "history": [{"assistant": a, "observation": o} for a, o in history],
            "episode_steps": env.step_i, "wall_time": time.time() - t0,
            "build_ok": False, "sim_ok": False, "structural_ok": False,
            "physics_ok": False, "goal_ok": False, "multipoint_ok": False}


def _save(d, task, result, memory_text):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in result.items() if k != "final_text"}
    (d / f"{task['task_id']}.json").write_text(
        json.dumps({"task": {k: v for k, v in task.items() if k != "topology"},
                    "memory_text": memory_text,
                    "result": slim,
                    "final_text": result.get("final_text", "")}, indent=1))
