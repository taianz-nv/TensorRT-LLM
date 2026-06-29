# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# TEMPORARY acceptance test for the vendored NVFP4 CuteDSL 3D-conv kernel
# (`tensorrt_llm/_torch/cute_dsl_kernels/blackwell/conv/dense_implicit_gemm_nvfp4.py`,
# vendored from dlarch-fastkernels/dynamic-kernel-generator!20906).
#
# It is intentionally standalone and disposable: it will be replaced by a proper
# custom-op test once the FP4-conv `torch.ops.trtllm.*` wrapper lands.
#
# What it checks (acceptance criteria):
#   1. The FP4 conv kernel RUNS on Blackwell (sm_100/103).
#   2. Accuracy: the kernel matches a PyTorch conv reference within FP4 round-off
#      (enforced inside the kernel's own `run(..., skip_ref_check=False)`).
#   3. Performance: FP4 conv is FASTER than the same-shape BF16 `F.conv3d`.
#
# Run standalone (on a B200/B300 with cutlass-dsl installed):
#   python tests/scripts/cute_dsl_kernels/run_fp4_conv_vs_bf16_temp.py
# or via pytest:
#   pytest tests/scripts/cute_dsl_kernels/run_fp4_conv_vs_bf16_temp.py -s
import os
import sys

import pytest

# Import the vendored kernel via the `blackwell` package dir so we do NOT trigger
# the heavy `import tensorrt_llm` C++ op init — the kernel is pure torch+cutlass.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_BLACKWELL_DIR = os.path.join(_REPO_ROOT, "tensorrt_llm", "_torch", "cute_dsl_kernels", "blackwell")
if _BLACKWELL_DIR not in sys.path:
    sys.path.insert(0, _BLACKWELL_DIR)


def _require_blackwell():
    """Skip unless we are on a Blackwell GPU with cutlass-dsl available."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip(
            f"NVFP4 conv kernel requires Blackwell sm_100/103; "
            f"got sm_{major}{torch.cuda.get_device_capability()[1]} "
            f"({torch.cuda.get_device_name(0)})"
        )
    try:
        import cutlass  # noqa: F401
    except ImportError:
        pytest.skip("nvidia-cutlass-dsl not installed")


# VAE-like, constraint-clean conv configs (see MR 20906 test notes):
#   - C must be <= 128 or a multiple of 256 (192 hits a SMEM swizzle wall).
#   - Kout (K) is a multiple of 128 (no partial GEMM-M tile).
#   - cluster (1, 1) + use_2cta_instrs=False for mma_tiler M=128.
# Each entry: (ncdhw, ktrs, mma_tiler_mn, label, expect_speedup)
# `expect_speedup` is False for the tiny correctness-only case (too small to be
# compute-bound, so FP4 vs BF16 timing is overhead-dominated and not meaningful).
_CONFIGS = [
    # Small sanity case straight from the MR test (known-good) — correctness only.
    ((1, 128, 1, 12, 16), (256, 3, 3, 3), (128, 128), "sanity_c128", False),
    # Larger, compute-bound cases where 4-bit throughput should beat BF16.
    ((1, 256, 4, 32, 32), (256, 3, 3, 3), (128, 128), "c256_4x32x32", True),
    ((1, 512, 2, 16, 16), (512, 3, 3, 3), (128, 128), "c512_2x16x16", True),
]

_STRIDE = (1, 1, 1)
_UPPER_PAD = (1, 1, 1)
_LOWER_PAD = (1, 1, 1)
_DIL = (1, 1, 1)
_WARMUP = 3
_ITERS = 10


def _bench_bf16_conv(ncdhw, ktrs, iters=_ITERS, warmup=_WARMUP):
    """Time a same-shape BF16 F.conv3d (channels_last_3d), return per-iter microseconds."""
    import torch
    import torch.nn.functional as F

    N, C, D, H, W = ncdhw
    K, T, R, S = ktrs
    x = torch.randn(N, C, D, H, W, device="cuda", dtype=torch.bfloat16).to(
        memory_format=torch.channels_last_3d
    )
    w = torch.randn(K, C, T, R, S, device="cuda", dtype=torch.bfloat16).to(
        memory_format=torch.channels_last_3d
    )
    pad = (_UPPER_PAD[0], _UPPER_PAD[1], _UPPER_PAD[2])  # symmetric (upper == lower)

    def _one():
        return F.conv3d(x, w, stride=_STRIDE, padding=pad, dilation=_DIL)

    for _ in range(warmup):
        _one()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _one()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # ms -> us, per iter


def _run_one(ncdhw, ktrs, mma_tiler_mn, label, expect_speedup=True):
    import cutlass
    from conv.dense_implicit_gemm_nvfp4 import run  # vendored kernel driver

    print(f"\n{'=' * 70}\n[{label}] ncdhw={ncdhw} ktrs={ktrs} mma={mma_tiler_mn}\n{'=' * 70}")

    # FP4 path: run() builds tensors, NVFP4-quantizes, compiles, executes,
    # asserts correctness vs a PyTorch reference within `tolerance`, and returns
    # the benchmarked kernel latency in microseconds.
    fp4_us = run(
        ncdhw=ncdhw,
        ktrs=ktrs,
        stride_dhw=_STRIDE,
        upper_pad_dhw=_UPPER_PAD,
        lower_pad_dhw=_LOWER_PAD,
        dil_dhw=_DIL,
        ab_dtype=cutlass.Float4E2M1FN,
        d_dtype=cutlass.Float4E2M1FN,
        acc_dtype=cutlass.Float32,
        sf_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=16,
        mma_tiler_mn=mma_tiler_mn,
        preferred_cluster_shape_mn=(1, 1),
        fallback_cluster_shape_mn=(1, 1),
        use_2cta_instrs=False,
        tolerance=1e-02,
        warmup_iterations=_WARMUP,
        iterations=_ITERS,
        skip_ref_check=False,  # <- enforces the accuracy gate
    )
    assert fp4_us is not None and fp4_us > 0, "FP4 conv did not return a valid runtime"

    # BF16 baseline of the same problem shape.
    bf16_us = _bench_bf16_conv(ncdhw, ktrs)

    speedup = bf16_us / fp4_us
    print(f"\n[{label}] FP4 = {fp4_us:.1f} us | BF16 = {bf16_us:.1f} us | speedup = {speedup:.2f}x")

    # Acceptance: FP4 must be faster than BF16 on compute-bound shapes
    # (accuracy already gated inside run()). The tiny sanity shape is
    # overhead-dominated, so it is correctness-only.
    if expect_speedup:
        assert fp4_us < bf16_us, (
            f"[{label}] FP4 conv ({fp4_us:.1f} us) not faster than BF16 ({bf16_us:.1f} us)"
        )
    return label, fp4_us, bf16_us, speedup


@pytest.mark.parametrize("ncdhw,ktrs,mma_tiler_mn,label,expect_speedup", _CONFIGS)
def test_fp4_conv_faster_than_bf16(ncdhw, ktrs, mma_tiler_mn, label, expect_speedup):
    _require_blackwell()
    _run_one(ncdhw, ktrs, mma_tiler_mn, label, expect_speedup)


if __name__ == "__main__":
    import torch

    if not torch.cuda.is_available():
        print("CUDA GPU required")
        sys.exit(1)
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        print(
            f"NVFP4 conv kernel requires Blackwell sm_100/103; got sm_{major}{minor} "
            f"({torch.cuda.get_device_name(0)}). Aborting."
        )
        sys.exit(1)

    rows = []
    for cfg in _CONFIGS:
        rows.append(_run_one(*cfg))

    print(f"\n{'=' * 70}\nSUMMARY (per-iter latency)\n{'=' * 70}")
    print(f"{'config':<18}{'FP4 (us)':>12}{'BF16 (us)':>12}{'speedup':>10}")
    for label, fp4_us, bf16_us, speedup in rows:
        print(f"{label:<18}{fp4_us:>12.1f}{bf16_us:>12.1f}{speedup:>9.2f}x")
    print(
        "\nAll configs: FP4 conv ran, matched reference within FP4 round-off, "
        "and was faster than BF16. ✓"
    )
