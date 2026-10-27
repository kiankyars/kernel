"""Actual symbolic tool runs and fail-closed evidence parsing."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from arena.safety import _result_evidence, _native_summary_clean, check_memory
from arena.spec import CandidateSpec


def test_symbolic_tool_checks_real_kernel_and_rejects_oob(tmp_path):
    result = check_memory(asdict(CandidateSpec()), tmp_path, cases=3)
    assert result["passed"], result
    probe, checked = result["checks"]
    assert probe["expected_rejection"] and probe["exit_code"] != 0
    assert checked["memory_cases_checked"] == 3
    assert checked["sanitizer_launches"] >= 3
    assert checked["numerical_cases_checked"] == 0


def test_bare_script_cannot_certify_or_run_oob_probe():
    script = Path(__file__).resolve().parents[1] / "scripts/check_memory.py"
    env = os.environ.copy()
    env["TRITON_INTERPRET"] = "1"
    child = subprocess.run([sys.executable, str(script), "--probe-oob"], env=env,
                           capture_output=True, text=True, timeout=30)
    assert child.returncode != 0
    assert "requires active tile-sanitizer" in child.stderr
    assert "ARENA_OOB_PROBE=armed" not in child.stdout


@pytest.mark.parametrize("stdout", [
    "", "ARENA_MEMORY_RESULT={invalid}", "ARENA_MEMORY_RESULT=[]",
    'ARENA_MEMORY_RESULT={"passed":true,"memory_cases_checked":3}',
    'ARENA_MEMORY_RESULT={"passed":true,"cases_run":3}',
])
def test_success_exit_without_tool_evidence_is_insufficient(stdout):
    assert _result_evidence(stdout, 3, symbolic=True) is None


def test_valid_evidence_distinguishes_numerical_from_symbolic():
    evidence = {"passed": True, "memory_cases_checked": 3, "numerical_cases_checked": 0,
                "sanitizer": "tilelens-symbolic", "sanitizer_active": True,
                "sanitizer_launches": 3}
    stdout = "ARENA_MEMORY_RESULT=" + json.dumps(evidence)
    assert _result_evidence(stdout, 3, symbolic=True) == evidence
    assert _result_evidence(stdout, 3, symbolic=False) is None
    assert _result_evidence(stdout, 0, symbolic=True) is None
    evidence["sanitizer_launches"] = 0
    assert _result_evidence("ARENA_MEMORY_RESULT=" + json.dumps(evidence), 3, symbolic=True) is None


def test_zero_memory_cases_rejected(tmp_path):
    with pytest.raises(ValueError, match="positive case count"):
        check_memory(asdict(CandidateSpec()), tmp_path, cases=0)


def test_native_summary_parsers_keep_tools_distinct():
    assert _native_summary_clean("memcheck", "========= ERROR SUMMARY: 0 errors")
    assert _native_summary_clean("racecheck", "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)")
    assert not _native_summary_clean("racecheck", "========= ERROR SUMMARY: 0 errors")
    assert not _native_summary_clean("racecheck", "========= RACECHECK SUMMARY: 1 hazards displayed (0 errors, 1 warnings)")
    assert not _native_summary_clean("memcheck", "========= ERROR SUMMARY: 1 errors")
