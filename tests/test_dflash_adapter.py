"""CPU checks for scoped patching and the real-model runner's refusal gates.

These tests do not claim to execute pretrained DFlash2 on CUDA.
"""

import importlib.util
import json
from pathlib import Path
import types

import pytest
import torch

from arena import dflash_adapter
from arena.spec import CandidateSpec


PINNED_FUNCTION = '''
def _grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.view(batch, length, groups, group_size)
    dynamic = dynamic.view(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        kernel = base[offset].view(1, 1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, offset], values)
    return output.view_as(hidden)
'''


@pytest.fixture
def module(tmp_path):
    path = tmp_path / "fake_dflash.py"
    path.write_text("import torch\nimport torch.nn.functional as F\n" + PINNED_FUNCTION)
    spec = importlib.util.spec_from_file_location("fake_dflash", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_dflash.py"
    spec = importlib.util.spec_from_file_location("benchmark_dflash", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensors():
    return torch.ones(1, 3, 4), torch.ones(1, 3, 2, 2), torch.ones(2, 4), 2


def test_pinned_ast_contract_and_reject_drift(module, tmp_path):
    assert dflash_adapter.validate_upstream(module) == dflash_adapter.PINNED_FUNCTION_AST_SHA256
    path = Path(module.__file__)
    path.write_text(path.read_text().replace("kernel * values", "kernel + values"))
    with pytest.raises(ValueError, match="differs from pinned"):
        dflash_adapter.validate_upstream(module)


def test_inference_patch_calls_candidate_and_restores(module, monkeypatch):
    import arena.kernels as kernels

    original = module._grouped_dynamic_convolve
    calls = []

    def candidate(hidden, dynamic, base, group_size, spec):
        assert not any(t.requires_grad for t in (hidden, dynamic, base))
        calls.append(spec)
        return original(hidden, dynamic, base, group_size)

    monkeypatch.setattr(kernels, "convolve", candidate)
    args = list(tensors())
    args[2] = torch.nn.Parameter(args[2])
    expected = original(*args)
    with dflash_adapter.patched_convolution(module, CandidateSpec()) as state:
        with torch.inference_mode():
            actual = module._grouped_dynamic_convolve(*args)
        assert state["calls"] == 1
        torch.testing.assert_close(actual, expected)
    assert module._grouped_dynamic_convolve is original
    assert calls == [CandidateSpec()]


def test_exception_restores_patch(module):
    original = module._grouped_dynamic_convolve
    with pytest.raises(RuntimeError, match="scope failed"):
        with dflash_adapter.patched_convolution(module, CandidateSpec()):
            raise RuntimeError("scope failed")
    assert module._grouped_dynamic_convolve is original


def test_training_call_rejected_and_restored(module):
    original = module._grouped_dynamic_convolve
    with pytest.raises(RuntimeError, match="inference-only"):
        with dflash_adapter.patched_convolution(module, CandidateSpec()):
            module._grouped_dynamic_convolve(*tensors())
    assert module._grouped_dynamic_convolve is original


def test_invalid_output_shape_rejected_and_restored(module, monkeypatch):
    import arena.kernels as kernels

    original = module._grouped_dynamic_convolve
    monkeypatch.setattr(kernels, "convolve", lambda hidden, *args: hidden[:, :1])
    with pytest.raises(RuntimeError, match="invalid DFlash"):
        with torch.inference_mode(), dflash_adapter.patched_convolution(module, CandidateSpec()):
            module._grouped_dynamic_convolve(*tensors())
    assert module._grouped_dynamic_convolve is original


def test_cpu_runner_fails_before_local_paths_or_optional_imports(runner, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "result.json"
    code = runner.main(["--target", "does-not-exist", "--draft", "does-not-exist",
                        "--prompts", "does-not-exist", "--candidate", "does-not-exist",
                        "--output", str(output)])
    assert code == 1
    assert "requires CUDA hardware" in capsys.readouterr().err
    assert not output.exists()


def test_interpreter_cannot_be_timed_as_cuda(runner, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    with pytest.raises(RuntimeError, match="Unset TRITON_INTERPRET"):
        runner.require_cuda()


def test_generation_gate_checks_tokens_and_acceptance(runner):
    baseline = {"output_ids": [[1, 2, 3]], "acceptance_lengths": [2]}
    runner._same_generation(baseline, dict(baseline))
    with pytest.raises(RuntimeError, match="token IDs differ"):
        runner._same_generation(baseline, {**baseline, "output_ids": [[1, 2, 4]]})
    with pytest.raises(RuntimeError, match="acceptance lengths differ"):
        runner._same_generation(baseline, {**baseline, "acceptance_lengths": [1, 1]})


@pytest.mark.parametrize("obj", [
    {"prompts": [], "max_new_tokens": 3},
    {"prompts": [[True]], "max_new_tokens": 3},
    {"prompts": [[-1]], "max_new_tokens": 3},
    {"prompts": [[1]], "max_new_tokens": 1},
    {"prompts": [[1]], "max_new_tokens": 3, "extra": 1},
])
def test_bad_workloads_rejected(runner, tmp_path, obj):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        runner.read_workload(path)


def test_local_checkpoint_manifest_detects_mutation(runner, tmp_path):
    (tmp_path / "config.json").write_text('{"architectures":["DFlash2DraftModel"]}')
    (tmp_path / "model.safetensors").write_bytes(b"fixture bytes; never loaded as a model")
    manifest = runner.checkpoint_manifest(tmp_path)
    assert len(manifest["files"]) == 2
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])
    runner._unchanged(manifest)
    (tmp_path / "model.safetensors").write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        runner._unchanged(manifest)
