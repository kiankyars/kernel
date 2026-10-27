"""Run actual memory tools with timeouts and explicit success evidence."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import re


def _result_evidence(stdout, cases, *, symbolic):
    """Fail closed on missing/malformed evidence or a zero-test tool invocation."""
    markers = [line.split("=", 1)[1] for line in stdout.splitlines()
               if line.startswith("ARENA_MEMORY_RESULT=")]
    if len(markers) != 1 or cases <= 0:
        return None
    try:
        evidence = json.loads(markers[0])
    except (ValueError, TypeError):
        return None
    if not isinstance(evidence, dict) or evidence.get("passed") is not True:
        return None
    if type(evidence.get("memory_cases_checked")) is not int or evidence["memory_cases_checked"] != cases:
        return None
    if symbolic:
        if (evidence.get("sanitizer") != "tilelens-symbolic"
                or evidence.get("sanitizer_active") is not True
                or type(evidence.get("sanitizer_launches")) is not int
                or evidence["sanitizer_launches"] < cases
                or evidence.get("numerical_cases_checked") != 0):
            return None
    elif (evidence.get("sanitizer") != "compute-sanitizer"
          or evidence.get("numerical_cases_checked") != cases):
        return None
    return evidence


def _native_summary_clean(tool, log):
    if tool == "memcheck":
        return re.search(r"ERROR SUMMARY: 0 errors\b", log) is not None
    if tool == "racecheck":
        return re.search(r"RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)", log) is not None
    return False


def check_memory(spec_dict, output_dir, *, device="cpu", seed=481516, cases=6):
    if device not in ("cpu", "cuda") or type(cases) is not int or cases <= 0:
        raise ValueError("memory checks require cpu/cuda and a positive case count")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[1] / "scripts/check_memory.py"
    arguments = [str(script), "--spec", json.dumps(spec_dict), "--device", device,
                 "--seed", str(seed), "--cases", str(cases)]
    environment = os.environ.copy()
    environment["TRITON_INTERPRET"] = "1" if device == "cpu" else "0"
    if device == "cpu":
        environment["ENABLE_SANITIZER"] = "1"
        executable = Path(sys.executable).parent / "tile-sanitizer"
        commands = [
            ("tilelens-negative-probe", [str(executable), str(script), "--probe-oob"]),
            ("tilelens", [str(executable), *arguments]),
        ]
    else:
        executable = shutil.which("compute-sanitizer")
        if executable is None:
            return {"passed": False, "status": "unavailable", "error": "compute-sanitizer is required"}
        commands = [(name, [executable, "--tool", name, "--error-exitcode", "86",
                            sys.executable, *arguments]) for name in ("memcheck", "racecheck")]
    results = []
    for name, command in commands:
        try:
            child = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"passed": False, "status": "tool_failure", "error": str(exc), "checks": results}
        log = child.stdout + "\n" + child.stderr
        path = output_dir / f"{name}.log"
        path.write_text(log)
        if name == "tilelens-negative-probe":
            passed = (child.returncode != 0 and "ARENA_OOB_PROBE=armed" in log
                      and "ARENA_OOB_PROBE=undetected" not in log
                      and re.search(r"out[- ]of[- ]bounds|ILLEGAL MEMORY ACCESS DETECTED",
                                    log, re.IGNORECASE) is not None)
            results.append({"tool": name, "passed": bool(passed),
                            "exit_code": child.returncode, "expected_rejection": True,
                            "log": str(path)})
            if not passed:
                return {"passed": False, "status": "negative_probe_failed", "checks": results}
            continue
        evidence = _result_evidence(child.stdout, cases, symbolic=device == "cpu")
        passed = child.returncode == 0 and evidence is not None
        # NVIDIA must confirm that the tool actually ran and saw no errors.
        if device != "cpu":
            passed = passed and _native_summary_clean(name, log)
        results.append({"tool": name, "passed": bool(passed), "exit_code": child.returncode,
                        "memory_cases_checked": evidence["memory_cases_checked"] if evidence else 0,
                        "numerical_cases_checked": evidence["numerical_cases_checked"] if evidence else 0,
                        "sanitizer_launches": evidence.get("sanitizer_launches") if evidence else None,
                        "log": str(path)})
        if not passed:
            return {"passed": False, "status": "failed", "checks": results}
    return {"passed": True, "status": "passed", "checks": results,
            "scope": "symbolic CPU bounds; not compiled GPU race safety" if device == "cpu" else "compiled CUDA memcheck and racecheck"}
