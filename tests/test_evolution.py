"""Controller tests mock execution, preserving real manifests and SQLite writes."""

from dataclasses import asdict
import json
import sqlite3
import stat
from types import SimpleNamespace

import pytest
import torch

from arena import benchmark, evolution, inference, oracle, safety
from arena.spec import CandidateSpec


@pytest.fixture
def arena_run(monkeypatch, tmp_path):
    state = {
        "correctness_fail": {"broken_shift"},
        "memory_fail": set(),
        "integration_fail": set(),
        "calls": {"correctness": [], "memory": [], "integration": [], "microbenchmark": []},
        "worker_payloads": [],
        "holdouts": [],
        "micro_speedup": 1e6,
        "integration_speedup": 1e6,
    }
    args = SimpleNamespace(
        output=tmp_path / "run", backend="interpret", population=3,
        generations=1, workers=1, cases=5, repeats=1, seed=98123,
    )

    class InlinePool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, function, payloads):
            payloads = list(payloads)
            state["worker_payloads"].extend(payloads)
            return [function(payload) for payload in payloads]

    def holdout_seed(bits):
        generation = len(state["holdouts"])
        submission_path = args.output / f"generation-{generation}-submissions.json"
        # Test chronology: candidates are frozen before the hidden seed exists.
        assert submission_path.is_file()
        assert stat.S_IMODE(submission_path.stat().st_mode) == 0o444
        submissions = json.loads(submission_path.read_text())["submissions"]
        assert len(submissions) == args.population
        assert all(entry["source_hash"] for entry in submissions)
        seed = 6000000000000000001 + generation
        state["holdouts"].append(seed)
        return seed

    def make_cases(seed, count, device):
        assert seed == state["holdouts"][-1]
        assert count == args.cases
        return [object() for _ in range(count)]

    def evaluate(spec, cases):
        state["calls"]["correctness"].append(spec.variant)
        passed = spec.variant not in state["correctness_fail"]
        return {"passed": passed, "cases_run": len(cases), "failures": [] if passed else ["negative control"]}

    def memory(spec, output_dir, **kwargs):
        state["calls"]["memory"].append(spec["variant"])
        return {"passed": spec["variant"] not in state["memory_fail"], "scope": "mocked"}

    def integration(spec, **kwargs):
        state["calls"]["integration"].append(spec.variant)
        passed = spec.variant not in state["integration_fail"]
        return {"passed": passed, "speedup": state["integration_speedup"] if passed else None}

    def micro(spec, seed, **kwargs):
        state["calls"]["microbenchmark"].append(spec.variant)
        return {"geometric_speedup": state["micro_speedup"]}

    monkeypatch.setattr(evolution, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(evolution.secrets, "randbits", holdout_seed)
    monkeypatch.setattr(oracle, "make_cases", make_cases)
    monkeypatch.setattr(oracle, "evaluate", evaluate)
    monkeypatch.setattr(safety, "check_memory", memory)
    monkeypatch.setattr(inference, "integration_check", integration)
    monkeypatch.setattr(benchmark, "microbenchmark", micro)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda: "mock GPU; no hardware measurements")
    return args, state


def test_failed_correctness_and_memory_never_reach_timing(arena_run):
    args, state = arena_run
    state["memory_fail"].add("split")
    report = evolution.run(args)
    assert state["calls"]["correctness"] == ["fused", "split", "broken_shift"]
    assert state["calls"]["memory"] == ["fused", "split"]
    assert state["calls"]["integration"] == ["fused"]
    assert state["calls"]["microbenchmark"] == ["fused"]
    statuses = {row["spec"]["variant"]: row["status"] for row in report["candidates"]}
    assert statuses == {"fused": "cpu_validated", "split": "memory_rejected", "broken_shift": "correctness_rejected"}
    for row in report["candidates"]:
        if row["status"].endswith("rejected"):
            assert row["microbenchmark"] is None
            assert row["gpu_fitness"] is None


def test_integration_failure_blocks_microbenchmark(arena_run):
    args, state = arena_run
    state["integration_fail"].add("fused")
    report = evolution.run(args)
    assert state["calls"]["microbenchmark"] == ["split"]
    fused = next(row for row in report["candidates"] if row["spec"]["variant"] == "fused")
    assert fused["status"] == "integration_rejected"
    assert fused["gpu_fitness"] is None


def test_cpu_speed_never_becomes_gpu_fitness(arena_run):
    args, state = arena_run
    state["micro_speedup"] = state["integration_speedup"] = 1e12
    report = evolution.run(args)
    assert report["gpu_winner"] is None
    assert not report["production_eligible"]
    assert all(row["gpu_fitness"] is None for row in report["candidates"])
    assert all(not row["production_eligible"] for row in report["candidates"])


def test_gpu_micro_only_improvement_cannot_win(arena_run):
    args, state = arena_run
    args.backend = "cuda"
    state["micro_speedup"], state["integration_speedup"] = 1000., 0.5
    report = evolution.run(args)
    assert report["gpu_winner"] is None
    survivors = [row for row in report["candidates"] if row["status"] == "gpu_provisional"]
    assert len(survivors) == 2
    assert all(row["gpu_fitness"] == 0.5 for row in survivors)
    assert not report["production_eligible"]


def test_worker_payloads_have_specs_and_mutation_seeds_only(arena_run):
    args, state = arena_run
    args.population, args.generations = 4, 2
    report = evolution.run(args)
    assert len(state["holdouts"]) == 2
    assert len(state["worker_payloads"]) == 5
    for parents, mutation_seed in state["worker_payloads"]:
        assert isinstance(mutation_seed, int)
        assert mutation_seed not in state["holdouts"]
        assert isinstance(parents, list)
        for parent in parents:
            assert set(parent) == set(asdict(CandidateSpec()))
            assert parent["variant"] != "broken_shift"
            CandidateSpec(**parent)
    for holdout in state["holdouts"]:
        assert str(holdout) not in json.dumps(state["worker_payloads"])
    replay = args.output / "replay-seeds.private.json"
    assert stat.S_IMODE(replay.stat().st_mode) == 0o600
    assert json.loads(replay.read_text()) == {str(i): seed for i, seed in enumerate(state["holdouts"])}
    assert report["evaluated"] == 8


def test_manifests_results_and_database_agree(arena_run):
    args, state = arena_run
    report = evolution.run(args)
    assert json.loads((args.output / "report.json").read_text()) == report
    with sqlite3.connect(args.output / "arena.sqlite") as database:
        rows = database.execute("SELECT candidate_id, source_hash, spec_json, result_json FROM evaluations").fetchall()
    assert len(rows) == report["evaluated"] == 3
    for candidate_id, source_hash, spec_json, result_json in rows:
        persisted = json.loads(result_json)
        assert persisted == next(row for row in report["candidates"] if row["candidate_id"] == candidate_id)
        assert persisted == json.loads((args.output / candidate_id / "result.json").read_text())
        assert source_hash == persisted["source_hash"]
        assert json.loads(spec_json) == persisted["spec"]
    submissions = json.loads((args.output / "generation-0-submissions.json").read_text())
    seal = json.loads((args.output / "generation-0-seal.json").read_text())
    assert seal["submission_sha256"] == evolution.digest(submissions)
    assert seal["seed_commitment"] == evolution.digest(state["holdouts"][0])


def test_nonempty_output_is_preserved(arena_run):
    args, state = arena_run
    args.output.mkdir()
    existing = args.output / "existing-result.json"
    existing.write_text('{"keep": true}')
    with pytest.raises(ValueError, match="not empty"):
        evolution.run(args)
    assert existing.read_text() == '{"keep": true}'
    assert list(args.output.iterdir()) == [existing]
    assert not state["calls"]["correctness"]


def test_mutation_is_bounded_repeatable_and_changes_one_field():
    parent = asdict(CandidateSpec())
    for seed in range(128):
        mutated = evolution.mutate(([parent], seed))
        assert mutated == evolution.mutate(([parent], seed))
        CandidateSpec(**mutated)
        assert set(mutated) == set(parent)
        assert sum(mutated[key] != parent[key] for key in parent) == 1
        assert mutated["variant"] in {"fused", "split"}


def test_microbenchmark_checks_new_workload_before_warmup_or_timing(monkeypatch):
    calls = []

    def wrong(hidden, dynamic, base, group_size, spec):
        calls.append("candidate")
        return torch.zeros_like(hidden)

    def expected(hidden, dynamic, base, group_size):
        calls.append("reference")
        return torch.ones_like(hidden)

    def forbidden_clock():
        raise AssertionError("Timer must not run for an incorrect candidate")

    monkeypatch.setattr(benchmark, "convolve", wrong)
    monkeypatch.setattr(benchmark, "reference", expected)
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", forbidden_clock)
    with pytest.raises(AssertionError, match="not close"):
        benchmark.microbenchmark(CandidateSpec(), seed=19, repeats=1)
    assert calls == ["candidate", "reference"]


def test_source_change_prevents_completed_report(arena_run, monkeypatch, tmp_path):
    args, state = arena_run
    # Exercise provenance against an isolated synthetic evaluator tree; never
    # rewrite the real shared checkout while its other tests/workers execute.
    synthetic = tmp_path / "evaluator"
    source = synthetic / "arena"
    scripts = synthetic / "scripts"
    source.mkdir(parents=True)
    scripts.mkdir()
    marker = source / "evolution.py"
    marker.write_text("# original evaluator\n")
    (scripts / "check_memory.py").write_text("# memory evaluator\n")
    monkeypatch.setattr(evolution, "__file__", str(marker))
    previous_evaluate = oracle.evaluate

    def changed_source(spec, cases):
        marker.write_text("# changed after submissions were frozen\n")
        return previous_evaluate(spec, cases)

    monkeypatch.setattr(oracle, "evaluate", changed_source)
    with pytest.raises(RuntimeError, match="source changed"):
        evolution.run(args)
    assert not (args.output / "report.json").exists()
    assert (args.output / "source/evolution.py").read_text() == "# original evaluator\n"
