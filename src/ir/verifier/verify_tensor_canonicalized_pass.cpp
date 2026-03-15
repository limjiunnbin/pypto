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

class TensorCanonicalizedVerifier : public IRVisitor {
 public:
  explicit TensorCanonicalizedVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

 protected:
  void VisitExpr_(const CallPtr& op) override {
    if (!op) return;
    auto global_var = std::dynamic_pointer_cast<const GlobalVar>(op->op_);
    if (global_var) {
      IRVisitor::VisitExpr_(op);
      return;
    }

    auto& registry = OpRegistry::GetInstance();
    if (!registry.IsRegistered(op->op_->name_)) {
      IRVisitor::VisitExpr_(op);
      return;
    }
    const auto& entry = registry.GetEntry(op->op_->name_);
    if (entry.GetOpCategory() != "TensorOp") {
      IRVisitor::VisitExpr_(op);
      return;
    }

    // Canonical tensor form expects no nested call arguments.
    for (const auto& arg : op->args_) {
      if (As<Call>(arg)) {
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "TensorCanonicalized", 0,
                                  "Tensor op has nested call argument: '" + op->op_->name_ + "'",
                                  op->span_);
      }
    }

    // tensor.slice canonical form uses (input, shape_tuple, offset_tuple[, valid_shape_tuple]).
    if (op->op_->name_ == "tensor.slice") {
      if (op->args_.size() >= 3) {
        if (!As<MakeTuple>(op->args_[1]) || !As<MakeTuple>(op->args_[2])) {
          diagnostics_.emplace_back(DiagnosticSeverity::Error, "TensorCanonicalized", 1,
                                    "tensor.slice expects tuple-form shape/offset arguments", op->span_);
        }
      }
    }

    IRVisitor::VisitExpr_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
};

class TensorCanonicalizedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "TensorCanonicalized"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      if (func->func_type_ != FunctionType::InCore) continue;
      TensorCanonicalizedVerifier verifier(diagnostics);
      verifier.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateTensorCanonicalizedPropertyVerifier() {
  return std::make_shared<TensorCanonicalizedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
