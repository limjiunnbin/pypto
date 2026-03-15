# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FlashAttention-style Tensor-IR demo block."""

import math

import pypto.language as pl
import torch
import torch.nn as nn


def fa_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reference FlashAttention-style forward pass."""
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(d))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


class TorchFABlock(nn.Module):
    """Original PyTorch FlashAttention-style block."""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return fa_reference(q, k, v)


def build_fa_program(seq_len: int = 16, hidden_dim: int = 64):
    """Build a Tensor-IR-first FA demo program."""
    scale = 1.0 / math.sqrt(float(hidden_dim))

    @pl.program
    class FAProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def fa_kernel(
            self,
            q: pl.Tensor[[seq_len, hidden_dim], pl.FP32],
            k_t: pl.Tensor[[hidden_dim, seq_len], pl.FP32],
            v: pl.Tensor[[seq_len, hidden_dim], pl.FP32],
        ) -> pl.Tensor[[seq_len, hidden_dim], pl.FP32]:
            scores: pl.Tensor[[seq_len, seq_len], pl.FP32] = pl.matmul(q, k_t)
            scaled: pl.Tensor[[seq_len, seq_len], pl.FP32] = pl.mul(scores, scale)
            row_max: pl.Tensor[[seq_len, 1], pl.FP32] = pl.row_max(scaled)
            shifted: pl.Tensor[[seq_len, seq_len], pl.FP32] = pl.row_expand_sub(scaled, row_max)
            exp_scores: pl.Tensor[[seq_len, seq_len], pl.FP32] = pl.exp(shifted)
            row_sum: pl.Tensor[[seq_len, 1], pl.FP32] = pl.row_sum(exp_scores)
            probs: pl.Tensor[[seq_len, seq_len], pl.FP32] = pl.row_expand_div(exp_scores, row_sum)
            out: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.matmul(probs, v)
            return out

        @pl.function
        def main(
            self,
            q: pl.Tensor[[seq_len, hidden_dim], pl.FP32],
            k_t: pl.Tensor[[hidden_dim, seq_len], pl.FP32],
            v: pl.Tensor[[seq_len, hidden_dim], pl.FP32],
        ) -> pl.Tensor[[seq_len, hidden_dim], pl.FP32]:
            out: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = self.fa_kernel(q, k_t, v)
            return out

    return FAProgram
