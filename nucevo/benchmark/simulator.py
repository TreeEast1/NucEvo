"""Self-contained OpenModelica simulation driver for NucEvo-Bench v1.

Compiles and simulates Modelica models with real `omc` in isolated temporary
directories, using the bundled `NuclearEdu.mo` library and content-addressed
result caching.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from nucevo.config import LIBRARY_DIR, cache_subdir, resolve_omc_binary

LIB_DIR = LIBRARY_DIR
LIBRARY_FILES = ["NuclearEdu.mo"]
OMC = resolve_omc_binary()


def _sim_env() -> dict[str, str]:
    env = dict(os.environ)
    omc_path = resolve_omc_binary()
    if omc_path.parent.exists():
        env["PATH"] = f"{omc_path.parent}:{env.get('PATH', '')}"
        lib_dir = omc_path.parent.parent / "lib"
        if lib_dir.exists():
            env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    om_lib = os.environ.get("OPENMODELICALIBRARY")
    if om_lib:
        env["OPENMODELICALIBRARY"] = om_lib
    env.setdefault("MAKEFLAGS", "-j8")
    return env


@dataclass
class SimResult:
    ok: bool
    load_ok: bool
    build_ok: bool
    sim_ok: bool
    messages: str
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    final: dict = field(default_factory=dict)
    series: dict = field(default_factory=dict)
    time: list = field(default_factory=list)
    wall_time: float = 0.0
    cached: bool = False
    raw_stdout: str = ""
    result_file: str = ""

    def to_json(self) -> dict:
        return dict(self.__dict__)


def _clean(messages: str) -> str:
    return "\n".join(
        line.strip()
        for line in messages.strip().splitlines()
        if line.strip() not in ("", "true", "false")
    )


def _write_files(directory: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def simulate(
    files: dict[str, str],
    top: str,
    stop_time: float = 600.0,
    n_intervals: int = 600,
    variables: list[str] | None = None,
    tolerance: float = 1e-6,
    with_series: bool = True,
    timeout: float = 60.0,
    extra_simflags: str = "",
    method: str = "dassl",
) -> SimResult:
    """Compile and simulate `top` from `files` (filename -> Modelica source)."""
    t0 = time.time()
    variables = variables or []
    try:
        lib_sig = "".join((LIB_DIR / name).read_text(encoding="utf-8") for name in LIBRARY_FILES)
    except OSError:
        lib_sig = ""
    key_src = json.dumps(
        {
            "files": files,
            "top": top,
            "stop": stop_time,
            "n": n_intervals,
            "lib": lib_sig,
            "vars": variables,
            "tol": tolerance,
            "s": with_series,
            "m": method,
            "f": extra_simflags,
        },
        sort_keys=True,
    )
    key = hashlib.sha1(key_src.encode()).hexdigest()[:20]
    cache_dir = cache_subdir("sim") / key
    cache_file = cache_dir / "result.json"
    if not with_series and cache_file.exists():
        try:
            cached_res = SimResult(**json.loads(cache_file.read_text(encoding="utf-8")))
            cached_res.cached = True
            return cached_res
        except Exception:
            pass

    tmp_parent = cache_subdir("tmp")
    work = Path(tempfile.mkdtemp(prefix="omsim_", dir=str(tmp_parent)))
    try:
        for lib_name in LIBRARY_FILES:
            shutil.copy(LIB_DIR / lib_name, work / lib_name)
        _write_files(work, files)
        mos = [f'loadFile("{work / LIBRARY_FILES[0]}"); a:=getErrorString();']
        for name in files:
            if name.endswith(".mo"):
                mos.append(f'loadFile("{work / name}"); a2:=getErrorString();')
        mos.append(f"buildModel({top}); b:=getErrorString();")
        mos.append(
            f'sim:=simulate({top}, stopTime={stop_time}, numberOfIntervals={n_intervals}, '
            f'tolerance={tolerance}, method="{method}", outputFormat="csv", '
            f'fileNamePrefix="{top}_sim"{extra_simflags}); c:=getErrorString();'
        )
        mos.append('print("===LOAD==="); print(a);')
        for name in files:
            if name.endswith(".mo"):
                mos.append("print(a2);")
        mos.append('print("===BUILD==="); print(b);')
        mos.append('print("===SIM==="); print(c);')
        (work / "run.mos").write_text("\n".join(mos), encoding="utf-8")

        omc_bin = resolve_omc_binary()
        timed_out = False
        proc = subprocess.Popen(
            [str(omc_bin), "run.mos"],
            cwd=work,
            env=_sim_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, _stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            stdout, _stderr = proc.communicate()

        if timed_out:
            return SimResult(
                False,
                False,
                False,
                False,
                f"omc/simulation timeout after {timeout:g}s",
                ["timeout"],
                [],
                wall_time=time.time() - t0,
            )

        match = re.search(r'messages = "(.*?)"\n', stdout, re.S)
        sim_msgs = match.group(1).replace("\\n", "\n").replace('\\"', '"') if match else ""
        load_ok = "===LOAD===" in stdout
        build_ok = "===BUILD===" in stdout and "Failed to build model" not in stdout
        sim_ok = build_ok and "The simulation finished successfully" in (sim_msgs + stdout)

        segments = {}
        for tag in ("LOAD", "BUILD", "SIM"):
            parts = stdout.split(f"==={tag}===", 1)
            segments[tag] = parts[1] if len(parts) > 1 else ""
        diag = segments["LOAD"] + segments["BUILD"] + segments["SIM"] + "\n" + sim_msgs
        errs = [line.strip() for line in diag.splitlines() if line.strip().startswith("Error")]
        warns = [
            line.strip()
            for line in diag.splitlines()
            if "Warning" in line and "not balanced" not in line
        ]

        final: dict[str, float] = {}
        for line in segments["SIM"].splitlines():
            if "=" in line and not line.startswith(("LOG", "Error", "Warning", "|")):
                var_name, _, raw_val = line.partition("=")
                try:
                    final[var_name] = float(raw_val.split()[0])
                except ValueError:
                    pass

        series: dict[str, list[float]] = {}
        times: list[float] = []
        candidates = sorted(work.glob("*_sim_res.csv")) + sorted(work.glob("*_res.csv"))
        if sim_ok and candidates:
            with candidates[0].open(encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = [col.strip().strip('"') for col in next(reader)]
                rows = list(reader)
            col_idx = {col: idx for idx, col in enumerate(header)}
            if "time" in col_idx:
                times = [float(row[col_idx["time"]]) for row in rows]
            wanted = variables or [col for col in header if col != "time"]
            if not with_series:
                wanted = [col for col in wanted if col != "time"]
                last_row = rows[-1] if rows else []
                if last_row:
                    for col in wanted:
                        if col in col_idx:
                            try:
                                final[col] = float(last_row[col_idx[col]])
                            except ValueError:
                                pass
                wanted = []
            for col in wanted:
                if col in col_idx:
                    try:
                        series[col] = [float(row[col_idx[col]]) for row in rows]
                    except ValueError:
                        series[col] = [float("nan")] * len(rows)

        if variables and series:
            for col in variables:
                if col in series and series[col]:
                    final[col] = series[col][-1]

        result = SimResult(
            ok=sim_ok,
            load_ok=load_ok,
            build_ok=build_ok,
            sim_ok=sim_ok,
            messages=_clean(sim_msgs + "\n" + segments["SIM"]),
            errors=errs,
            warnings=warns,
            final=final,
            series=series,
            time=times,
            wall_time=time.time() - t0,
            cached=False,
            raw_stdout=stdout[-8000:],
            result_file=str(candidates[0]) if candidates else "",
        )
        if not with_series and sim_ok:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                slim = SimResult(**{**result.__dict__, "series": {}, "time": [], "raw_stdout": ""})
                cache_file.write_text(json.dumps(slim.to_json()), encoding="utf-8")
            except OSError:
                pass
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)
