# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tensor-IR-first demo models for accuracy and lowering validation."""

from .fa_demo import TorchFABlock, build_fa_program, fa_reference
from .gdn_demo import TorchGDNBlock, build_gdn_program, gdn_reference
from .glm_demo import TorchGLMBlock, build_glm_program, glm_reference
from .mamba3_demo import TorchMamba3Block, build_mamba3_program, mamba3_reference

__all__ = [
    "TorchFABlock",
    "TorchMamba3Block",
    "TorchGDNBlock",
    "TorchGLMBlock",
    "fa_reference",
    "mamba3_reference",
    "gdn_reference",
    "glm_reference",
    "build_fa_program",
    "build_mamba3_program",
    "build_gdn_program",
    "build_glm_program",
]
