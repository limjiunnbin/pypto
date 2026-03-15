# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Accuracy tests for Tensor-IR demo modules against original PyTorch modules."""

import pytest
import torch
from pypto import passes

from examples.language.tensor_ir import (
    TorchFABlock,
    TorchGDNBlock,
    TorchGLMBlock,
    TorchMamba3Block,
    build_fa_program,
    build_gdn_program,
    build_glm_program,
    build_mamba3_program,
    fa_reference,
    gdn_reference,
    glm_reference,
    mamba3_reference,
)


def _run_tensor_ir_stage(program):
    """Run the explicit Tensor-IR stage then lower to tile IR."""
    program = passes.convert_to_ssa()(program)
    program = passes.flatten_call_expr()(program)
    program = passes.outline_incore_scopes()(program)
    program = passes.outline_cluster_scopes()(program)
    program = passes.canonicalize_tensor_ops()(program)
    program = passes.infer_tensor_shape_layout()(program)
    program = passes.plan_tensor_fusion()(program)
    program = passes.validate_tensor_lowerability()(program)
    program = passes.lower_tensor_to_tile()(program)
    return program


def test_fa_demo_accuracy_vs_original_pytorch():
    torch.manual_seed(42)
    q = torch.randn(16, 64)
    k = torch.randn(16, 64)
    v = torch.randn(16, 64)

    module = TorchFABlock()
    out_module = module(q, k, v)
    out_ref = fa_reference(q, k, v)
    assert torch.allclose(out_module, out_ref, rtol=1e-5, atol=1e-5)


def test_mamba3_demo_accuracy_vs_original_pytorch():
    torch.manual_seed(42)
    x = torch.randn(16, 64)
    module = TorchMamba3Block(hidden_dim=64)

    out_module = module(x)
    out_ref = mamba3_reference(x, module.w_in, module.w_gate, module.w_state)
    assert torch.allclose(out_module, out_ref, rtol=1e-5, atol=1e-5)


def test_gdn_demo_accuracy_vs_original_pytorch():
    torch.manual_seed(42)
    x = torch.randn(16, 64)
    module = TorchGDNBlock(hidden_dim=64)

    out_module = module(x)
    out_ref = gdn_reference(x, module.w_gate, module.w_delta)
    assert torch.allclose(out_module, out_ref, rtol=1e-5, atol=1e-5)


def test_glm_demo_accuracy_vs_original_pytorch():
    torch.manual_seed(42)
    x = torch.randn(16, 64)
    module = TorchGLMBlock(hidden_dim=64)

    out_module = module(x)
    out_ref = glm_reference(x, module.w_q, module.w_k, module.w_v, module.w_ff1, module.w_ff2)
    assert torch.allclose(out_module, out_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "builder",
    [
        build_fa_program,
        build_mamba3_program,
        build_gdn_program,
        build_glm_program,
    ],
)
def test_demo_programs_run_tensor_ir_stage_and_lower(builder):
    """Each demo program should pass through Tensor-IR stage and lower successfully."""
    program = builder(seq_len=16, hidden_dim=64)
    lowered = _run_tensor_ir_stage(program)
    assert lowered is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
