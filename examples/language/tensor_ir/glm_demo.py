# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""GLM-style Tensor-IR demo block."""

import math

import pypto.language as pl
import torch
import torch.nn as nn


def glm_reference(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_ff1: torch.Tensor,
    w_ff2: torch.Tensor,
) -> torch.Tensor:
    """Reference GLM-style transformer block."""
    d = x.shape[-1]
    q = torch.matmul(x, w_q)
    k = torch.matmul(x, w_k)
    v = torch.matmul(x, w_v)

    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(d))
    probs = torch.softmax(scores, dim=-1)
    ctx = torch.matmul(probs, v)

    ff1 = torch.matmul(ctx, w_ff1)
    act = ff1 * torch.sigmoid(ff1)
    ff2 = torch.matmul(act, w_ff2)
    return x + ctx + ff2


class TorchGLMBlock(nn.Module):
    """Original PyTorch GLM-style block."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w_q = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        self.w_k = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        self.w_v = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        self.w_ff1 = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        self.w_ff2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return glm_reference(x, self.w_q, self.w_k, self.w_v, self.w_ff1, self.w_ff2)


def build_glm_program(seq_len: int = 16, hidden_dim: int = 64):
    """Build a Tensor-IR-first GLM demo program."""
    scale = 1.0 / math.sqrt(float(hidden_dim))

    @pl.program
    class GLMProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def glm_kernel(
            self,
            x: pl.Tensor[[seq_len, hidden_dim], pl.FP32],
            w_q: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
            w_k: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
            w_v: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
            w_ff1: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
            w_ff2: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
        ) -> pl.Tensor[[seq_len, hidden_dim], pl.FP32]:
            q: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.matmul(x, w_q)
            k: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.matmul(x, w_k)
            v: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.matmul(x, w_v)

            scores: pl.Tensor[[seq_len, seq_len], pl.FP32] = pl.matmul(q, k, b_trans=True)
            scaled: pl.Tensor[[seq_len, seq_len], pl.FP32] = pl.mul(scores, scale)
            row_max: pl.Tensor[[seq_len, 1], pl.FP32] = pl.row_max(scaled)
            shifted: pl.Tensor[[seq_len, seq_len], pl.FP32] = pl.row_expand_sub(scaled, row_max)
            exp_scores: pl.Tensor[[seq_len, seq_len], pl.FP32] = pl.exp(shifted)
            row_sum: pl.Tensor[[seq_len, 1], pl.FP32] = pl.row_sum(exp_scores)
            probs: pl.Tensor[[seq_len, seq_len], pl.FP32] = pl.row_expand_div(exp_scores, row_sum)
            ctx: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.matmul(probs, v)

            ff1: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.matmul(ctx, w_ff1)
            neg_ff1: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.mul(ff1, -1.0)
            exp_neg: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.exp(neg_ff1)
            denom: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.add(exp_neg, 1.0)
            sig_ff1: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.recip(denom)
            act: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.mul(ff1, sig_ff1)
            ff2: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.matmul(act, w_ff2)

            tmp: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.add(x, ctx)
            out: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.add(tmp, ff2)
            return out

        @pl.function
        def main(
            self,
            x: pl.Tensor[[seq_len, hidden_dim], pl.FP32],
            w_q: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
            w_k: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
            w_v: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
            w_ff1: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
            w_ff2: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
        ) -> pl.Tensor[[seq_len, hidden_dim], pl.FP32]:
            out: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = self.glm_kernel(x, w_q, w_k, w_v, w_ff1, w_ff2)
            return out

    return GLMProgram
