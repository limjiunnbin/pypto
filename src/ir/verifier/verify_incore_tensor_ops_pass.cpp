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

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/verifier/verification_error.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

class IncoreTensorOpsVerifier : public IRVisitor {
 public:
  explicit IncoreTensorOpsVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

 protected:
  void VisitExpr_(const CallPtr& op) override {
    if (!op) return;

    // Ignore function calls.
    if (std::dynamic_pointer_cast<const GlobalVar>(op->op_)) {
      IRVisitor::VisitExpr_(op);
      return;
    }

    auto& registry = OpRegistry::GetInstance();
    if (!registry.IsRegistered(op->op_->name_)) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "IncoreTensorOps", 0,
                                "Unregistered op found in InCore function: '" + op->op_->name_ + "'",
                                op->span_);
      IRVisitor::VisitExpr_(op);
      return;
    }

    static const std::unordered_set<std::string> kAllowedCategories = {
        "TensorOp", "TileOp", "SyncOp", "CrossCoreOp"};
    const auto& entry = registry.GetEntry(op->op_->name_);
    if (!kAllowedCategories.count(entry.GetOpCategory())) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "IncoreTensorOps", 1,
                                "Unsupported op category '" + entry.GetOpCategory() + "' for op '" +
                                    op->op_->name_ + "' in InCore function",
                                op->span_);
    }

    IRVisitor::VisitExpr_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
};

class IncoreTensorOpsPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "IncoreTensorOps"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      if (func->func_type_ != FunctionType::InCore) continue;
      IncoreTensorOpsVerifier verifier(diagnostics);
      verifier.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateIncoreTensorOpsPropertyVerifier() {
  return std::make_shared<IncoreTensorOpsPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
