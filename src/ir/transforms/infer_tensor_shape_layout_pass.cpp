/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "pypto/ir/transforms/passes.h"

#include "pypto/ir/program.h"
#include "pypto/ir/transforms/pass_properties.h"

namespace pypto {
namespace ir {
namespace pass {

Pass InferTensorShapeLayout() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // Current stage implementation is metadata-preserving. Explicit inference
    // rewrites can be added incrementally without changing pipeline contracts.
    return program;
  };
  return CreateProgramPass(pass_func, "InferTensorShapeLayout", kInferTensorShapeLayoutProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
