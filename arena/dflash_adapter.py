"""Scoped inference-only replacement for the pinned public DFlash2 operation.

Use in a single-threaded benchmark process. This patches a module global and is
not a production multi-request dispatch mechanism. No optional DFlash package
is imported until the caller supplies its module.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import inspect
import textwrap
import threading

import torch

from .spec import CandidateSpec


DFLASH_COMMIT = "07ebd93db9f472af339b644bb70221ad8428328a"
PINNED_FUNCTION_AST_SHA256 = "3a2f0619417d8f8b79d7da86efb86612c44c5899b5d2634bc34e0ef84de446d4"
_PATCH_LOCK = threading.RLock()


def function_ast_digest(function) -> str:
    """Hash syntax, ignoring only whitespace, comments and the function name."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, SyntaxError) as exc:
        raise ValueError("DFlash convolution source is unavailable for verification") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("expected one inspectable DFlash convolution function")
    tree.body[0].name = "_grouped_dynamic_convolve"
    return hashlib.sha256(ast.dump(tree.body[0], include_attributes=False).encode()).hexdigest()


def validate_upstream(module) -> str:
    """Refuse silent replacement of a function with different semantics."""
    original = getattr(module, "_grouped_dynamic_convolve", None)
    digest = function_ast_digest(original)
    if digest != PINNED_FUNCTION_AST_SHA256:
        raise ValueError(
            "DFlash convolution source differs from pinned commit " + DFLASH_COMMIT
            + "; review its contract before using this adapter"
        )
    return digest


@contextmanager
def patched_convolution(module, spec: CandidateSpec):
    """Patch only this scope, restoring the original even after an exception.

    Calls must occur under torch.inference_mode(). Detached input views permit
    frozen inference over Parameter slices without providing an autograd path.
    The yielded counter proves real inference traversed the candidate kernel.
    """
    if not isinstance(spec, CandidateSpec):
        raise ValueError("spec must be a CandidateSpec")
    with _PATCH_LOCK:
        validate_upstream(module)
        original = module._grouped_dynamic_convolve
        state = {"calls": 0, "input_elements": 0}

        def replacement(hidden, dynamic, base, group_size):
            if not torch.is_inference_mode_enabled():
                raise RuntimeError("DFlash kernel replacement is inference-only; use torch.inference_mode()")
            from .kernels import convolve

            result = convolve(hidden.detach(), dynamic.detach(), base.detach(), group_size, spec)
            if (not isinstance(result, torch.Tensor) or result.shape != hidden.shape
                    or result.dtype != hidden.dtype or result.device != hidden.device):
                raise RuntimeError("candidate returned an invalid DFlash convolution output")
            state["calls"] += 1
            state["input_elements"] += hidden.numel()
            return result

        module._grouped_dynamic_convolve = replacement
        try:
            yield state
        finally:
            module._grouped_dynamic_convolve = original
