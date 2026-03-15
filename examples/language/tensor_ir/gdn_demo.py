# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""GDN-style Tensor-IR demo block."""

import pypto.language as pl
import torch
import torch.nn as nn


def gdn_reference(x: torch.Tensor, w_gate: torch.Tensor, w_delta: torch.Tensor) -> torch.Tensor:
    """Reference GDN-style gated delta update."""
    gate = torch.sigmoid(torch.matmul(x, w_gate))
    delta = torch.matmul(x, w_delta)
    return x + gate * delta


class TorchGDNBlock(nn.Module):
    """Original PyTorch GDN-style block."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        self.w_delta = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gdn_reference(x, self.w_gate, self.w_delta)


def build_gdn_program(seq_len: int = 16, hidden_dim: int = 64):
    """Build a Tensor-IR-first GDN demo program."""

    @pl.program
    class GDNProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def gdn_kernel(
            self,
            x: pl.Tensor[[seq_len, hidden_dim], pl.FP32],
            w_gate: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
            w_delta: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
        ) -> pl.Tensor[[seq_len, hidden_dim], pl.FP32]:
            gate_logits: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.matmul(x, w_gate)
            neg_logits: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.mul(gate_logits, -1.0)
            exp_neg: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.exp(neg_logits)
            denom: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.add(exp_neg, 1.0)
            gate: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.recip(denom)
            delta: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.matmul(x, w_delta)
            gated_delta: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.mul(gate, delta)
            out: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = pl.add(x, gated_delta)
            return out

        @pl.function
        def main(
            self,
            x: pl.Tensor[[seq_len, hidden_dim], pl.FP32],
            w_gate: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
            w_delta: pl.Tensor[[hidden_dim, hidden_dim], pl.FP32],
        ) -> pl.Tensor[[seq_len, hidden_dim], pl.FP32]:
            out: pl.Tensor[[seq_len, hidden_dim], pl.FP32] = self.gdn_kernel(x, w_gate, w_delta)
            return out

    return GDNProgram
