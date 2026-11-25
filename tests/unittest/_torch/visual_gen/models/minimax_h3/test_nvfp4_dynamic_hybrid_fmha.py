# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for MiniMax-H3 NVFP4 dynamic-scale hybrid FMHA (qkasym + early dynamic cuDNN)."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

NVFP4_DIR = (
    Path(__file__).resolve().parents[6]
    / "tensorrt_llm/_torch/visual_gen/models/minimax_h3/minimax_h3_nvfp4__sm_107__u1_u4"
)


class _Scalar:
    def __init__(self, v: float):
        self.v = float(v)

    def __mul__(self, other):
        return _Scalar(self.v * (other.v if isinstance(other, _Scalar) else other))

    def __rmul__(self, other):
        return _Scalar(other * self.v)

    def __truediv__(self, other):
        return _Scalar(self.v / (other.v if isinstance(other, _Scalar) else other))

    def __rtruediv__(self, other):
        return _Scalar(other / self.v)

    def clamp(self, min=None, max=None):
        x = self.v
        if min is not None:
            mv = min.v if isinstance(min, _Scalar) else min
            x = x if x >= mv else mv
        if max is not None:
            xv = max.v if isinstance(max, _Scalar) else max
            x = x if x <= xv else xv
        return _Scalar(x)

    def bfloat16(self):
        return self

    def float(self):
        return self


class _Vec:
    def __init__(self, vals):
        self.vals = [float(x) for x in vals]

    def float(self):
        return self

    def __truediv__(self, denom):
        return _Vec([x / denom for x in self.vals])

    def clamp_min(self, lo):
        return _Vec([x if x >= lo else lo for x in self.vals])

    def __getitem__(self, idx):
        return _Scalar(self.vals[idx])


def _load_complementary():
    tree = ast.parse((NVFP4_DIR / "kernels/fused_fp8_qk.py").read_text())
    keep = {"_FP16SM_K_SHIFT_SCALE", "complementary_fp16sm_descales", "_DYNAMIC_OVERRIDE", "set_dynamic_override"}
    body = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in keep)
        or (
            isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and n.target.id in keep
        )
    ]
    ns = {
        "math": math,
        "torch": SimpleNamespace(Tensor=object, stack=lambda xs: [x.v for x in xs]),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "fused_fp8_qk.py", "exec"), ns)
    return ns


class TestNvfp4DynamicHybridFmha(unittest.TestCase):
    def test_complementary_descales_unit_fold_and_tiny_k_guard(self):
        ns = _load_complementary()
        fn = ns["complementary_fp16sm_descales"]
        sm_scale = 128.0**-0.5
        for k_max in (6.0, 14.0, 40.0):
            dq, dk, dv = fn(_Vec([15.0, k_max, 18.0]), 128)
            self.assertAlmostEqual(dk, 32.0 * (k_max / 448.0), places=7)
            self.assertAlmostEqual(dv, 18.0 / 448.0, places=7)
            self.assertAlmostEqual(dq * dk * sm_scale * math.log2(math.e), 1.0, places=6)
        # Degenerate tiny K must not force Q underflow to zero
        dq, dk, _ = fn(_Vec([8.0, 0.01, 9.0]), 128)
        self.assertAlmostEqual(dq * dk * sm_scale * math.log2(math.e), 1.0, places=6)
        self.assertGreaterEqual(8.0 / dq, 0.125 - 1e-6)

    def test_dynamic_override_toggle(self):
        ns = _load_complementary()
        self.assertFalse(ns["_DYNAMIC_OVERRIDE"])
        ns["set_dynamic_override"](True)
        self.assertTrue(ns["_DYNAMIC_OVERRIDE"])
        ns["set_dynamic_override"](False)
        self.assertFalse(ns["_DYNAMIC_OVERRIDE"])

    def test_config_accepts_qkasym_and_hybrid_fields(self):
        tree = ast.parse((NVFP4_DIR / "config.py").read_text())
        src = (NVFP4_DIR / "config.py").read_text()
        self.assertIn('fp8_scale_fusion: str = "qkasym"', src)
        self.assertIn("attention_cudnn_steps: int = 8", src)
        self.assertIn("skip_corr_start_step: int = 25", src)
        self.assertIsNotNone(tree)


if __name__ == "__main__":
    unittest.main()
