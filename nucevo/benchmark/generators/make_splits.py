"""Build the frozen NucEvolve-Bench v1 splits.

Generation is rejection-sampled and then validated:

  1. the reference must simulate, be steady, and pass the reference-free physics
     checker and the full multi-operating-point verifier;
  2. every repair/tune task must actually be broken;
  3. signatures are globally unique across all splits (evolution first, then dev,
     then the test splits), so no test task can appear in the experience stream;
  4. no test task is ever generated from a template that evolution also uses,
     except the ID split, which by design shares templates but not tasks.

Outputs: benchmark/splits/<name>.jsonl plus MANIFEST.json with counts, hashes,
structural fingerprints and per-split family/template distributions.
"""
from __future__ import annotations

import collections
import hashlib
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nucevo.benchmark.generators import tasks as GT                     # noqa: E402
from nucevo.benchmark.generators import topologies as T                 # noqa: E402
from nucevo.benchmark.verifier import v3                                # noqa: E402

SPLITS = {
    "evolution": {
        "templates": T.EVOLUTION_TEMPLATES, "n": 600, "seed": 7001,
        "families": {"assemble": 0.28, "repair": 0.44, "tune": 0.18,
                     "diagnose": 0.05, "multistep": 0.05},
        "rename": None,
    },
    "dev": {
        "templates": T.EVOLUTION_TEMPLATES, "n": 160, "seed": 7002,
        "families": {"assemble": 0.30, "repair": 0.45, "tune": 0.15,
                     "diagnose": 0.05, "multistep": 0.05},
        "rename": None,
    },
    "id_test": {
        "templates": T.EVOLUTION_TEMPLATES, "n": 240, "seed": 7003,
        "families": {"assemble": 0.30, "repair": 0.45, "tune": 0.15,
                     "diagnose": 0.05, "multistep": 0.05},
        "rename": None,
    },
    "ood_near": {
        "templates": T.OOD_NEAR_TEMPLATES, "n": 240, "seed": 7004,
        "families": {"assemble": 0.40, "repair": 0.40, "tune": 0.10,
                     "diagnose": 0.05, "multistep": 0.05},
        "rename": None,
    },
    "ood_far": {
        "templates": T.OOD_FAR_TEMPLATES, "n": 200, "seed": 7005,
        "families": {"assemble": 0.40, "repair": 0.40, "tune": 0.10,
                     "diagnose": 0.05, "multistep": 0.05},
        "rename": None,
    },
    "stress": {
        "templates": T.EVOLUTION_TEMPLATES + T.OOD_NEAR_TEMPLATES, "n": 120,
        "seed": 7006,
        "families": {"repair": 0.40, "multistep": 0.40, "assemble": 0.20},
        "rename": "pool", "multi_fault": True,
    },
}


def _pick_family(rng, weights):
    r, acc = rng.random(), 0.0
    for k, w in weights.items():
        acc += w
        if r <= acc:
            return k
    return list(weights)[-1]


def _worker(job):
    """Generate one task and validate its reference with the full verifier."""
    template, family, seed, idx, split, rename, n_faults = job
    rng = random.Random(seed + idx * 7919)
    tid = f"{split[:3].upper()}_{template}_{family[0].upper()}_{idx:04d}"
    task = GT.try_make_task(tid, template, family, rng, split,
                            n_faults=n_faults, attempts=30,
                            rename_scheme=rename)
    if task is None:
        return None
    ref = v3.verify(task, GT.reference_text(task["topology"], task["overrides"]),
                    do_multipoint=True, do_transient=True)
    if not ref["success"]:
        return {"__invalid__": tid,
                "reason": {k: ref[k] for k in ("build_ok", "sim_ok", "steady",
                                               "structural_ok", "physics_ok",
                                               "goal_ok", "multipoint_ok",
                                               "transient_ok")}}
    task["reference_validation"] = {"v3_success": True,
                                    "multipoint": ref.get("multipoint_ok"),
                                    "transient": ref.get("transient_ok")}
    return task


def generate_split(name, cfg, seen, workers=32, oversample=1.25):
    n_target = cfg["n"]
    families = cfg["families"]
    templates = cfg["templates"]
    jobs = []
    rng = random.Random(cfg["seed"])
    i = 0
    # deterministic family/template schedule with a fixed seed
    while i < int(n_target * oversample) + 10:
        template = templates[i % len(templates)]
        family = _pick_family(rng, families)
        nf = 2 if (cfg.get("multi_fault") and family in ("repair", "multistep")) else 1
        jobs.append((template, family, cfg["seed"], i, name, cfg.get("rename"), nf))
        i += 1
    out, invalid = [], []
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for j, res in enumerate(pool.imap_unordered(_worker, jobs, chunksize=2)):
            if res is None:
                continue
            if "__invalid__" in res:
                invalid.append(res)
                continue
            sig = res["signature"]
            if sig in seen:
                continue
            seen.add(sig)
            out.append(res)
            if len(out) >= n_target:
                break
            if (j + 1) % 25 == 0:
                print(f"  {name}: {len(out)}/{n_target} ({time.time()-t0:.0f}s)",
                      flush=True)
    return out, invalid


def main(workers=32, only=None, scale=1.0):
    splits_dir = HERE / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    manifest = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "library_sha1": hashlib.sha1(
                    (HERE / "nuclear_edu" / "NuclearEdu.mo").read_bytes()).hexdigest(),
                "scale": scale, "splits": {}}
    for name, cfg in SPLITS.items():
        if only and name not in only:
            continue
        cfg = dict(cfg)
        cfg["n"] = max(4, int(cfg["n"] * scale))
        if scale != 1.0:
            cfg["seed"] = cfg["seed"] + 500
        print(f"=== {name} === target {cfg['n']}", flush=True)
        tasks, invalid = generate_split(name, cfg, seen, workers)
        p = splits_dir / f"{name}.jsonl"
        with open(p, "w") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")
        h = hashlib.sha1(p.read_bytes()).hexdigest()
        fam = collections.Counter(t["family"] for t in tasks)
        topo = collections.Counter(t["template"] for t in tasks)
        fam_ok = collections.Counter(t["family"] for t in tasks if t["reference_validation"]["v3_success"])
        manifest["splits"][name] = {
            "n": len(tasks), "sha1": h, "path": str(p),
            "templates": dict(topo), "families": dict(fam),
            "reference_valid": len(tasks), "invalid_rejected": len(invalid),
            "structural_fingerprints": sorted({t["fingerprint"]["hash"] for t in tasks}),
            "n_structures": len({t["fingerprint"]["hash"] for t in tasks}),
        }
        print(f"  wrote {len(tasks)} tasks -> {p} sha1={h[:12]} "
              f"structures={len({t['fingerprint']['hash'] for t in tasks})}", flush=True)
    (splits_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1)[:2500])
    return manifest


if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else None
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    main(w, only, scale)
