"""A bounded mutation space for real Triton programs, never arbitrary Python."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CandidateSpec:
    """Kernel choices workers may mutate without changing the trusted oracle.

    ``split`` launches once per convolution offset and is the structural seed;
    ``fused`` keeps the accumulator in one kernel. ``broken_shift`` is a negative
    control which changes causal indexing and must never pass the oracle.
    """

    block_size: int = 128
    num_warps: int = 4
    unroll: bool = True
    variant: Literal["fused", "split", "broken_shift"] = "fused"

    def __post_init__(self) -> None:
        if type(self.block_size) is not int or self.block_size not in (32, 64, 128, 256):
            raise ValueError("block_size must be one of 32, 64, 128, 256")
        if type(self.num_warps) is not int or self.num_warps not in (1, 2, 4, 8):
            raise ValueError("num_warps must be one of 1, 2, 4, 8")
        if type(self.unroll) is not bool:
            raise ValueError("unroll must be a boolean")
        if self.variant not in ("fused", "split", "broken_shift"):
            raise ValueError("variant must be fused, split, or broken_shift")
