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

#include <algorithm>
#include <string>

#include "pypto/core/error.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/verifier/property_verifier_registry.h"

namespace pypto {
namespace ir {
namespace pass {

Pass ValidateTensorLowerability() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    auto& registry = PropertyVerifierRegistry::GetInstance();
    IRPropertySet props{IRProperty::TensorLowerable};
    auto diagnostics = registry.VerifyProperties(props, program);
    bool has_errors = std::any_of(diagnostics.begin(), diagnostics.end(),
                                  [](const Diagnostic& d) { return d.severity == DiagnosticSeverity::Error; });
    if (has_errors) {
      throw pypto::ValueError("Tensor lowerability validation failed:\n" +
                              PropertyVerifierRegistry::GenerateReport(diagnostics));
    }
    return program;
  };
  return CreateProgramPass(pass_func, "ValidateTensorLowerability", kValidateTensorLowerabilityProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
