# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Mamba3-style Tensor-IR demo block."""

import pypto.language as pl
import torch
import torch.nn as nn


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def mamba3_reference(
    x: torch.Tensor,
    w_in: torch.Tensor,
    w_gate: torch.Tensor,
    w_state: torch.Tensor,
) -> torch.Tensor:
    """Reference Mamba3-style gated state update."""
    x_proj = torch.matmul(x, w_in)
    gate = _sigmoid(torch.matmul(x, w_gate))
    state = torch.matmul(x, w_state)
    return gate * state + (1.0 - gate) * x_proj


class TorchMamba3Block(nn.Module):
    """Original PyTorch Mamba3-style block."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w_in = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        self.w_gate = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        self.w_state = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mamba3_reference(x, self.w_in, self.w_gate, self.w_state)


def build_mamba3_program(seq_len: int = 16, hidden_dim: int = 64):
    """Build a Tensor-IR-first Mamba3 demo program."""

    @pl.program
    class Mamba3Program:
        @pl.function(type=pl.FunctionType.InCore)
        def mamba3_kernel(
            self,
            x: pl.Tensor[[seq_len, hidden_dim], pl.FP32],
            w_in: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
            w_gate: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
            w_state: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
        ) -> pl.Tensor[[seq_len, hidden_dim], pl.FP32]:
            x_proj: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.matmul(x, w_in)
            gate_logits: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.matmul(x, w_gate)
            neg_logits: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.mul(gate_logits, -1.0)
            exp_neg: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.exp(neg_logits)
            denom: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.add(exp_neg, 1.0)
            gate: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.recip(denom)

            state: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.matmul(x, w_state)
            gated_state: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.mul(gate, state)

            one: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.expands(gate, 1.0)
            one_minus_gate: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.sub(one, gate)
            carry: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.mul(one_minus_gate, x_proj)

            out: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.add(gated_state, carry)
            return out

        @pl.function
        def main(
            self,
            x: pl.Tensor[[seq_len, hidden_dim], pl.FP32],
            w_in: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
            w_gate: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
            w_state: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
        ) -> pl.Tensor[[seq_len, hidden_dim], pl.FP32]:
            out: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = self.mamba3_kernel(x, w_in, w_gate, w_state)
            return out

    return Mamba3Program
