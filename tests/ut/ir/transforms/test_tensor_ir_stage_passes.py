# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for Tensor-IR stage passes."""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes


def _make_program():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
            return y

        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
            return y

    return Program


def test_tensor_stage_pipeline_sequence_runs():
    """Tensor stage passes run successfully before tile lowering."""
    program = _make_program()
    program = passes.convert_to_ssa()(program)
    program = passes.flatten_call_expr()(program)
    program = passes.outline_incore_scopes()(program)
    program = passes.outline_cluster_scopes()(program)
    program = passes.canonicalize_tensor_ops()(program)
    program = passes.infer_tensor_shape_layout()(program)
    program = passes.plan_tensor_fusion()(program)
    program = passes.validate_tensor_lowerability()(program)
    program = passes.lower_tensor_to_tile()(program)
    assert program is not None


def test_lower_tensor_to_tile_matches_legacy_convert_pass():
    """New lowering pass keeps behavior compatible with legacy pass."""
    program = _make_program()
    program = passes.convert_to_ssa()(program)
    program = passes.flatten_call_expr()(program)
    program = passes.outline_incore_scopes()(program)
    program = passes.outline_cluster_scopes()(program)
    program = passes.canonicalize_tensor_ops()(program)
    program = passes.infer_tensor_shape_layout()(program)
    program = passes.plan_tensor_fusion()(program)
    program = passes.validate_tensor_lowerability()(program)

    lowered = passes.lower_tensor_to_tile()(program)
    legacy = passes.convert_tensor_to_tile_ops()(program)
    ir.assert_structural_equal(lowered, legacy)


def test_validate_tensor_lowerability_reports_missing_conversion():
    """Validation fails early for TensorOp without lowering rule."""
    span = ir.Span.unknown()
    tensor_type = ir.TensorType([4], DataType.FP32)
    x_param = ir.Var("x", tensor_type, span)
    call = ir.create_op_call("test.tensor_op_no_conv", [x_param], {}, span)
    y_var = ir.Var("y", tensor_type, span)
    body = ir.SeqStmts([ir.AssignStmt(y_var, call, span), ir.ReturnStmt([y_var], span)], span)
    func = ir.Function("incore", [x_param], [tensor_type], body, span, ir.FunctionType.InCore)
    program = ir.Program([func], "test_program", span)

    with pytest.raises(Exception, match="Tensor lowerability validation failed"):
        passes.validate_tensor_lowerability()(program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
