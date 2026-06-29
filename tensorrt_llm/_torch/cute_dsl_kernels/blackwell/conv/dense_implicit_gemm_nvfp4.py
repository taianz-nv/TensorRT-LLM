# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Vendored from dlarch-fastkernels/dynamic-kernel-generator!20906 (feature/nvfp4-conv);
# cross-file imports rewritten to relative. Do not edit by hand — re-vendor from source.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
from typing import Optional, Tuple, Type, Union, Literal
from functools import lru_cache
import cuda.bindings.driver as cuda
import sys
import os

import torch
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
from cutlass.cute.runtime import from_dlpack
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.pipeline as pipeline
from dataclasses import dataclass
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineOp,
    PipelineState,
    PipelineAsync,
    agent_sync,
    pipeline_init_arrive,
    pipeline_init_wait,
)
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(current_dir, "../../../"))

from .dense_gemm_persistent_dynamic_preferred_cluster import (
    PersistentDenseGemmKernelDynamicPreferredCluster,
)


from cutlass.cutlass_dsl import if_generate as _if_generate
from cutlass._mlir.dialects import nvvm


@dataclass(frozen=True)
class PipelineTmaCpAsyncUmma(PipelineAsync):
    """
    Merged pipeline for TMA (A+B+SFB) producers AND cp.async (SFA) producers sharing a single mbarrier set.

    Full mbarrier uses BOTH independent mbarrier counters:
      - tx_count   : incremented via mbarrier.expect_tx for TMA bytes, decremented by TMA completion
      - arrive_cnt : initialized to cp.async producer group size; decremented by cp.async.mbarrier.arrive.noinc

    Barrier phase flips only when tx_count == 0 AND arrive_cnt == 0, so the UMMA consumer performs
    a single wait/release per stage covering both producer streams.

    General variant: supports arbitrary cluster shapes (including cluster_shape_n > 1).
    """

    is_leader_cta: bool
    cta_group: cute.nvgpu.tcgen05.CtaGroup
    tx_count: int

    @staticmethod
    def create(
        *,
        num_stages: int,
        cpasynd_producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: int,
        barrier_storage: cute.Pointer,
        cta_layout_vmnk: Optional[cute.Layout] = None,
        mcast_mode_mn: Tuple[int, int] = (1, 1),
        defer_sync: bool = False,
    ) -> "PipelineTmaCpAsyncUmma":
        """Creates and initializes a merged TMA+cp.async / UMMA pipeline."""
        if not isinstance(barrier_storage, cute.Pointer):
            raise ValueError(
                f"Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}"
            )

        producer = (PipelineOp.AsyncLoad, cpasynd_producer_group)
        consumer = (PipelineOp.TCGen05Mma, consumer_group)

        sync_object_full = pipeline.PipelineTmaUmma._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer, tx_count
        )
        sync_object_empty = pipeline.PipelineTmaUmma._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
        )

        cta_group = (
            cute.nvgpu.tcgen05.CtaGroup.ONE
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, mode=[0]) == 1
            else cute.nvgpu.tcgen05.CtaGroup.TWO
        )

        if cta_layout_vmnk is None or cute.size(cta_layout_vmnk) == 1:
            consumer_mask = None
            is_leader_cta = True
            # No cross-CTA; remote arrives degenerate to self-arrive.
            producer_mask = cutlass.Int32(0)
        else:
            # Compute the empty-drain multicast mask: OR over A-axis and
            # B-axis TMA multicast sets (plus their V-peers).
            # PipelineTmaUmma._compute_mcast_arrival_mask returns the same
            # value but invoking it here produces stale IR for the merged
            # sync_object_empty; inline the equivalent computation instead.
            cta_rank_here = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            coord_self = cta_layout_vmnk.get_flat_coord(cta_rank_here)
            coord_peer = (coord_self[0] ^ 1, *coord_self[1:])
            # Leader cluster rank (v=0 within this V-group). For V=2 the
            # flat rank is v + m*V + ..., so clearing the V bit yields the
            # leader. Both leader and peer's cpasync lanes remote-arrive on
            # leader's sfa_full barrier to close the peer-sSFA race.
            producer_mask = cta_rank_here & ~cutlass.Int32(1)
            mask_a_self = cute.nvgpu.cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, coord_self, mcast_mode=2
            )
            mask_b_self = cute.nvgpu.cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, coord_self, mcast_mode=1
            )
            mask_a_peer = cute.nvgpu.cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, coord_peer, mcast_mode=2
            )
            mask_b_peer = cute.nvgpu.cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, coord_peer, mcast_mode=1
            )
            if mcast_mode_mn[0] == 1 and mcast_mode_mn[1] == 1:
                consumer_mask = (
                    cutlass.Int32(mask_a_self)
                    | cutlass.Int32(mask_b_self)
                    | cutlass.Int32(mask_a_peer)
                    | cutlass.Int32(mask_b_peer)
                )
            elif mcast_mode_mn[1] == 1:
                consumer_mask = cutlass.Int32(mask_b_self) | cutlass.Int32(mask_b_peer)
            else:
                assert mcast_mode_mn[0] == 1
                consumer_mask = cutlass.Int32(mask_a_self) | cutlass.Int32(mask_a_peer)
            is_leader_cta = pipeline.PipelineTmaUmma._compute_is_leader_cta(
                cta_layout_vmnk
            )

        if not defer_sync:
            cute.arch.mbarrier_init_fence()
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk) == 1:
                agent_sync(Agent.ThreadBlock)
            else:
                agent_sync(Agent.ThreadBlockCluster, is_relaxed=True)

        return PipelineTmaCpAsyncUmma(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            consumer_mask,
            is_leader_cta,
            cta_group,
            tx_count,
        )

    def producer_acquire_tma(
        self,
        state: PipelineState,
        try_acquire_token: Optional[cutlass.Boolean] = None,
        *,
        expected_tx: Optional[cutlass.Int32] = None,
    ) -> None:
        """TMA producer path.

        Only the leader CTA issues ``arrive_and_expect_tx``. The multicast
        variant broadcasts BOTH the arrive and the tx-expect to every peer
        barrier in the cluster, so peer CTAs must do nothing here. Any peer
        ``arrive(1)`` would double-count the arrive on the peer barrier and
        opens a race window against cp.async producers draining first.
        """
        _if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )
        tx = self.tx_count if expected_tx is None else expected_tx

        def _leader_arrive_and_expect_tx() -> None:
            self.sync_object_full.arrive_and_expect_tx(state.index, tx)

        _if_generate(
            self.is_leader_cta,
            _leader_arrive_and_expect_tx,
        )

    def producer_acquire_cpasync(
        self,
        state: PipelineState,
        try_acquire_token: Optional[cutlass.Boolean] = None,
    ) -> None:
        """cp.async producer path: only wait for buffer empty; arrive is issued via producer_commit_cpasync."""
        _if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )

    def producer_commit_cpasync(self, state: PipelineState) -> None:
        """Each cp.async lane contributes one arrive via cp.async.mbarrier.arrive.noinc."""
        self.sync_object_full.arrive_cp_async_mbarrier(state.index)

    def consumer_release(self, state: PipelineState) -> None:
        """UMMA consumer releases the shared empty barrier once per stage (multicast to peer CTAs)."""
        self.sync_object_empty.arrive(state.index, self.consumer_mask, self.cta_group)

    # ------------------------------------------------------------------
    # Shift-by-K cross-CTA producer commit primitives.
    # In 2CTA clusters, cp.async.mbarrier.arrive.noinc only signals the
    # local CTA's sfa_full, so the peer's LDGSTS landing on leader SMEM
    # is not covered. Split the commit into commit_group + wait_group(K)
    # + cross-CTA mbarrier.arrive(dst=leader_rank) so that arrives are
    # deferred by K iters, giving LDGSTS time to land on both CTAs
    # before the leader MMA fires.
    def producer_cp_async_commit(self) -> None:
        """Register all prior LDGSTS as one cp.async group."""
        cute.arch.cp_async_commit_group()

    def producer_cp_async_wait(self, num_inflight: int) -> None:
        """Block until at most ``num_inflight`` cp.async groups remain pending."""
        cute.arch.cp_async_wait_group(num_inflight)

    def producer_arrive_remote(self, stage_index) -> None:
        """Remote-arrive on leader's sfa_full[stage_index] via DSMEM."""
        bar = self.sync_object_full.get_barrier(stage_index)
        cute.arch.mbarrier_arrive(bar, self.producer_mask)


"""
A high-performance 3D implicit-GEMM based fprop convolution example for the NVIDIA Blackwell SM100 architecture using CUTE DSL.
- Input tensor A is NxDxHxWxC, must be C major.
- Filter tensor B is KxTxRxSxC, must be C major.
- Output tensor C is NxZxPxQxK, must be K major.

This kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations and for the im2col transformation of the input tensor A.
    - Utilizes Blackwell's tcgen05 MMA for matrix multiply-accumulate (MMA) operations (including 2cta mma instructions)
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Utilizes either a dense GEMM kernel or a persistent dense GEMM kernel for the implicit GEMM.

This implicit-GEMM based convolution works by converting the convolution into a GEMM problem with the following mapping:
- GEMM M dimension maps to NxZxPxQ
- GEMM N dimension maps to K
- GEMM K dimension maps to TxRxSxC
During the load of input tensor to SMEM, the TMA operation performs the im2col transformation on the input tensor A.
This transforms the A matrix into the required shape for the GEMM operation (NxZxPxQ by TxRxSxC), and may involve replication of the input elements.
Filter tensor can be loaded to SMEM without any transformation.
The output tensor C is then stored to GMEM via TMASTG.im2col (no transformation necessary).

To run this example:

.. code-block:: bash

    python examples/blackwell/dense_implicit_gemm_nvfp4_ai.py         \
      --ncdhw 1,128,32,32,32 --ktrs 256,3,3,3                         \
      --use_2cta_instrs --mma_tiler_mn 256,128                        \
      --preferred_cluster_shape_mn 2,1 --fallback_cluster_shape_mn 1,1 \
      --upper_pad_dhw 1,1,1 --lower_pad_dhw 1,1,1                     \
      --stride_dhw 1,1,1 --dil_dhw 1,1,1

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/dense_implicit_gemm_nvfp4_ai.py     \
      --ncdhw 1,128,32,32,32 --ktrs 256,3,3,3                         \
      --use_2cta_instrs --mma_tiler_mn 256,128                        \
      --preferred_cluster_shape_mn 2,1 --fallback_cluster_shape_mn 1,1 \
      --upper_pad_dhw 1,1,1 --lower_pad_dhw 1,1,1                     \
      --stride_dhw 1,1,1 --dil_dhw 1,1,1                              \
      --warmup_iterations 1 --iterations 10 --skip_ref_check

Constraints:
* Supported input data types: fp16, bf16, tf32, int8, uint8, fp8 (e4m3fn, e5m2),
  see detailed valid dtype combinations in below ConvKernel class documentation
* A/B tensor must have the same data type
* Mma tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
* Mma tiler N must be 32-256, step 32
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if use_2cta_instrs=True
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 4, 8, and 16 for TFloat32,
  Float16/BFloat16, and Int8/Uint8/Float8, respectively.
"""


def _check_tensor_alignment(
    c: int,
    k: int,
    ab_dtype: Type[cutlass.Numeric],
    d_dtype: Type[cutlass.Numeric],
):
    """Check if the tensor alignment is valid for convolution.

    :param c: The number of input channels
    :type c: int
    :param k: The number of output channels
    :type k: int
    :param ab_dtype: The data type of the A and B operands
    :type ab_dtype: Type[cutlass.Numeric]
    :param d_dtype: The data type of the output tensor
    :type d_dtype: Type[cutlass.Numeric]
    """

    def check_contiguous_16B_alignment(dtype, num_major_elements):
        num_contiguous_elements = 16 * 8 // dtype.width
        return num_major_elements % num_contiguous_elements == 0

    if not check_contiguous_16B_alignment(
        d_dtype, k
    ) or not check_contiguous_16B_alignment(ab_dtype, c):
        raise testing.CantImplementError(
            f"Invalid tensor alignment: C = {c}, K = {k}, ab_dtype = {ab_dtype}, d_dtype = {d_dtype}"
        )


def _check_swizzle_size(
    m: int,
    n: int,
    mma_tiler_mn: Tuple[int, int],
    use_2cta_instrs: bool,
    preferred_cluster_shape_mn: Tuple[int, int],
    fallback_cluster_shape_mn: Tuple[int, int],
    swizzle_size: int,
    raster_along: str,
):
    """Check that swizzle_size does not exceed the cluster count in the swizzled dimension."""
    if swizzle_size <= 1:
        return

    cta_v = 2 if use_2cta_instrs else 1
    cta_tile_m = mma_tiler_mn[0] // cta_v
    cta_tile_n = mma_tiler_mn[1]
    m_tiles = -(-m // cta_tile_m)
    n_tiles = -(-n // cta_tile_n)

    for cs in [preferred_cluster_shape_mn, fallback_cluster_shape_mn]:
        if raster_along == "m":
            nclusters = -(-n_tiles // cs[1])
        else:
            nclusters = -(-m_tiles // cs[0])
        if nclusters < swizzle_size:
            dim_name = "N" if raster_along == "m" else "M"
            raise testing.CantImplementError(
                f"swizzle_size ({swizzle_size}) exceeds the number of "
                f"{dim_name} clusters ({nclusters}) for cluster shape "
                f"{cs}. Use a smaller swizzle_size or increase the "
                f"{dim_name} dimension."
            )


class Sm100BlockScaledPersistentDenseImplicitGemmKernel(
    PersistentDenseGemmKernelDynamicPreferredCluster
):
    """
    Persistent 3D convolution kernel.
    The input (A) is expected to be in 5D tensor (NDHWC) format and is loaded via TMALDG.im2col atom.
    The filter (B) is expected to be in 5D tensor (KTRSC) format and is loaded via TMALDG atom.
    The output (C) is expected to be in 5D tensor (NZPQK) format and is stored via TMASTG.im2col.
    This class reuses the kernel from the PersistentDenseGemmKernel class for the implicit GEMM.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param use_2cta_instrs: Whether to use CTA group 2 for advanced thread cooperation
    :type use_2cta_instrs: bool
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tiler (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]
    :param filter_trs: Filter dimensions (T, R, S)
    :type filter_trs: Tuple[int, int, int]
    :param upper_padding_dhw: Upper padding (PadD, PadH, PadW)
    :type upper_padding_dhw: Tuple[int, int, int]
    :param lower_padding_dhw: Lower padding (PadD, PadH, PadW)
    :type lower_padding_dhw: Tuple[int, int, int]
    :param stride_dhw: Stride (Sd, Sh, Sw)
    :type stride_dhw: Tuple[int, int, int]
    :param dilation_dhw: Dilation (DilD, DilH, DilW)
    :type dilation_dhw: Tuple[int, int, int]
    :param swizzle_size: Swizzling size in the unit of cluster. 1 means no swizzle
    :type swizzle_size: int
    :param raster_along: Rasterization order of clusters. Only used when swizzle_size > 1.
    :type raster_along: Literal["m", "n"]

    :note: In current version, A and B tensor must be C major. C tensor must be K major.

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported A/B data types:
        - TFloat32
        - Float16/BFloat16
        - Int8/Uint8
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulator data types:
        - Float32 (for all floating point A/B data types)
        - Float16 (only for fp16 and fp8 A/B data types)
        - Int32 (only for uint8/int8 A/B data types)

    :note: Supported C data types:
        - Float32 (for float32 and int32 accumulator data types)
        - Int32 (for float32 and int32 accumulator data types)
        - Float16/BFloat16 (for fp16 and fp8 accumulator data types)
        - Int8/Uint8 (for uint8/int8 accumulator data types)
        - Float8E4M3FN/Float8E5M2 (for float32 accumulator data types)

    :note: Constraints:
        - MMA tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
        - MMA tiler N must be 32-256, step 32
        - Cluster shape M must be multiple of 2 if use_2cta_instrs=True
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16

    **Example:**

    .. code-block:: python
        conv = Sm100BlockScaledPersistentDenseImplicitGemmKernel(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=True,
            mma_tiler_mn=(128, 128),
            preferred_cluster_shape_mn=(2, 2),
            fallback_cluster_shape_mn=(1, 1),
            filter_trs=(3, 3, 3),
            upper_padding_dhw=(1, 1, 1),
            lower_padding_dhw=(1, 1, 1),
            stride_dhw=(1, 1, 1),
            dilation_dhw=(1, 1, 1),
        )
        conv(a, b, d, stream, epilogue_op)
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        preferred_cluster_shape_mn: Tuple[int, int],
        fallback_cluster_shape_mn: Tuple[int, int],
        input_C: int,
        output_K: int,
        output_nzpq: Tuple[int, int, int, int],
        swizzle_size: int = 1,
        raster_along: Literal["m", "n"] = "m",
    ):
        super().__init__(
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=mma_tiler_mn,
            preferred_cluster_shape_mn=preferred_cluster_shape_mn,
            fallback_cluster_shape_mn=fallback_cluster_shape_mn,
            use_tma_store=True,  # Conv always uses im2col TMA store
            swizzle_size=swizzle_size,
            raster_along=raster_along,
        )
        self.sf_vec_size = sf_vec_size

        # Layout-shaping geometry must stay host Python int so it remains
        # trace-const: input_C feeds the mma_tiler K dim (SMEM shape), output_K
        # feeds SFB/SFD host sizing, and output_nzpq (N,Z,P,Q) shapes the SFD
        # global descriptor whose M-mode drives the SIMT autovec_copy store
        # fragment (must be statically shaped).
        self.input_C = input_C
        self.output_K = output_K
        self.output_nzpq = output_nzpq

        # Override warp specialization for fp4 conv: 4 LDGSTS_SFA warps + sched warp.
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.ldgsts_sfa_warp_id = (6, 7, 8, 9)
        self.sched_warp_id = 10
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_warp_id,
                *self.ldgsts_sfa_warp_id,
                self.sched_warp_id,
                *self.epilogue_warp_id,
            )
        )
        # Override barrier ids; parent's preferred_cluster init reserves bar 0 for cta_sync.
        self.epilog_sync_bar_id = 1
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilog_sync_bar_id,
            num_threads=32 * len(self.epilogue_warp_id),
        )
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3

    def _setup_conv_input_attrs(self, a, b, d):
        """Validate and set input-dependent attributes.

        Sets a_dtype, b_dtype, d_dtype, a_major_mode, b_major_mode, d_layout.
        """
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.d_dtype: Type[cutlass.Numeric] = d.element_type
        if cutlass.const_expr(a.leading_dim != 4):
            raise RuntimeError("The layout of a is not supported")
        if cutlass.const_expr(b.leading_dim != 4):
            raise RuntimeError("The layout of b is not supported")
        if cutlass.const_expr(d.leading_dim != 4):
            raise RuntimeError("The layout of d is not supported")
        self.a_major_mode = cute.nvgpu.OperandMajorMode.K
        self.b_major_mode = cute.nvgpu.OperandMajorMode.K
        self.d_layout = utils.LayoutEnum.ROW_MAJOR

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # SFD path: only supported for FP4 (NVFP4) output. Caller opts in by
        # passing both sfd_tensor and norm_const_tensor; gen_sfd is decided in
        # __call__ from those tensors. FP8/FP16/BF16/FP32 outputs do not carry
        # SFD (matches TRT-LLM dense_blockscaled_gemm convention).
        # sfd_dtype is bound to sf_dtype in __call__ (NVFP4 standard: E4M3).
        self.sfd_vec_size: int = 16
        self.M_D: float = 6.0 if self.d_dtype is cutlass.Float4E2M1FN else None

    def _setup_conv_tma(
        self, a, b, d, tiled_mma, upper_pad_op, lower_pad_op, stride_op, dil_op
    ):
        """Set up dual-cluster TMA atoms and tensors for im2col convolution.

        Creates preferred and fallback TMA atoms for A and B tensors, and a single
        TMA atom for D (im2col store is cluster-independent).

        upper_pad_op/lower_pad_op/stride_op/dil_op are the pad/stride/dilation
        operand tuples that build the im2col A descriptor corner. They are
        runtime Int32 tuples so one compiled cubin serves any pad/stride/dil
        config without recompilation.

        :returns: (tma_atom_a_preferred, tma_tensor_a_preferred,
                   tma_atom_a_fallback, tma_tensor_a_fallback,
                   tma_atom_b_preferred, tma_tensor_b_preferred,
                   tma_atom_b_fallback, tma_tensor_b_fallback,
                   tma_atom_d, tma_tensor_d)
        """
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Create 2-mode hierarchical tensor layout: (N, D, H, W, C) -> ((W, H, D, N), C)
        mA = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[3, 2, 1, 0, 4]))
        mA = cute.group_modes(mA, begin=0, end=4)

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        a_internal_type = (
            cutlass.TFloat32 if mA.element_type is cutlass.Float32 else None
        )

        # Filter T/R/S sourced from the filter tensor b (K, T, R, S, C) rather
        # than host ints. The filter tensor is marked dynamic-layout, so these
        # extents are runtime Int32 and one compiled cubin serves any T/R/S.
        rt_filter_trs = (b.shape[1], b.shape[2], b.shape[3])

        # --- A preferred ---
        a_copy_atom_preferred = (
            cpasync.CopyBulkTensorIm2ColG2SMulticastOp(cta_group=self.cta_group)
            if self.is_preferred_a_mcast
            else cpasync.CopyBulkTensorIm2ColG2SOp(cta_group=self.cta_group)
        )
        tma_atom_a_preferred, tma_tensor_a_preferred = (
            cute.nvgpu.make_im2col_tma_atom_A(
                a_copy_atom_preferred,
                mA,
                a_smem_layout,
                self.mma_tiler,
                tiled_mma,
                rt_filter_trs,
                upper_pad_op,
                lower_pad_op,
                stride_op,
                dil_op,
                self.preferred_cluster_layout_vmnk.shape,
                internal_type=a_internal_type,
            )
        )

        # --- A fallback ---
        a_copy_atom_fallback = (
            cpasync.CopyBulkTensorIm2ColG2SMulticastOp(cta_group=self.cta_group)
            if self.is_fallback_a_mcast
            else cpasync.CopyBulkTensorIm2ColG2SOp(cta_group=self.cta_group)
        )
        tma_atom_a_fallback, tma_tensor_a_fallback = cute.nvgpu.make_im2col_tma_atom_A(
            a_copy_atom_fallback,
            mA,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            rt_filter_trs,
            upper_pad_op,
            lower_pad_op,
            stride_op,
            dil_op,
            self.fallback_cluster_layout_vmnk.shape,
            internal_type=a_internal_type,
        )

        # --- B: tiled TMA load ---
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        # Filter (K, T, R, S, C) -> reorder to (K, (C, S, R, T)) for coord_iter indexing.
        mB = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[0, 4, 3, 2, 1]))
        mB = cute.group_modes(mB, begin=1, end=5)
        b_internal_type = (
            cutlass.TFloat32 if mB.element_type is cutlass.Float32 else None
        )

        # --- B preferred ---
        b_op_preferred = sm100_utils.cluster_shape_to_tma_atom_B(
            self.preferred_cluster_shape_mn, tiled_mma.thr_id
        )
        tma_atom_b_preferred, tma_tensor_b_preferred = cute.nvgpu.make_tiled_tma_atom_B(
            b_op_preferred,
            mB,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.preferred_cluster_layout_vmnk.shape,
            internal_type=b_internal_type,
        )

        # --- B fallback ---
        b_op_fallback = sm100_utils.cluster_shape_to_tma_atom_B(
            self.fallback_cluster_shape_mn, tiled_mma.thr_id
        )
        tma_atom_b_fallback, tma_tensor_b_fallback = cute.nvgpu.make_tiled_tma_atom_B(
            b_op_fallback,
            mB,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.fallback_cluster_layout_vmnk.shape,
            internal_type=b_internal_type,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size
        # CLC response size is 4B * 4 elements
        self.num_clc_response_bytes = 16

        # --- D: TMASTG im2col store (cluster-independent) ---
        mD = cute.make_tensor(d.iterator, cute.select(d.layout, mode=[3, 2, 1, 0, 4]))
        mD = cute.group_modes(mD, begin=0, end=4)

        epi_smem_layout = cute.slice_(
            self.d_smem_layout_staged, (None, None, (None, 0))
        )

        tma_atom_d, tma_tensor_d = cpasync.make_im2col_tma_atom(
            cpasync.CopyBulkTensorIm2ColS2GOp(),
            mD,
            epi_smem_layout,
            self.epi_tile,
        )
        tma_tensor_d = cute.coalesce(tma_tensor_d, target_profile=(1, 1))

        def add_dummy_batch_dimension(tensor):
            new_layout = cute.append(tensor.layout, cute.make_layout(1))
            return cute.make_tensor(tensor.iterator, new_layout)

        tma_tensor_a_preferred = add_dummy_batch_dimension(tma_tensor_a_preferred)
        tma_tensor_a_fallback = add_dummy_batch_dimension(tma_tensor_a_fallback)
        tma_tensor_b_preferred = add_dummy_batch_dimension(tma_tensor_b_preferred)
        tma_tensor_b_fallback = add_dummy_batch_dimension(tma_tensor_b_fallback)
        tma_tensor_d = add_dummy_batch_dimension(tma_tensor_d)

        return (
            tma_atom_a_preferred,
            tma_tensor_a_preferred,
            tma_atom_a_fallback,
            tma_tensor_a_fallback,
            tma_atom_b_preferred,
            tma_tensor_b_preferred,
            tma_atom_b_fallback,
            tma_tensor_b_fallback,
            tma_atom_d,
            tma_tensor_d,
        )

    def _setup_attributes(self):
        """Set up configurations that are dependent on convolution inputs."""
        # Compute mma instruction shapes
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        # Dynamic K tiling: when C < 4*mma_inst_shape_k (256), reduce K_gemm_tile
        # so each K tile fits within one filter position (C elements).
        # This ensures A's im2col RestK == B's flat RestK.
        mma_inst_tile_k = min(4, self.input_C // mma_inst_shape_k)
        # Expose for overlapping-accum SF column accounting (see below).
        self.mma_inst_tile_k = mma_inst_tile_k
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            (mma_inst_shape_k * mma_inst_tile_k,),
        )
        self.mma_tiler_sfa = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k // 16,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        print(f"[tiler] mma_tiler     = {self.mma_tiler}")
        print(f"[tiler] mma_tiler_sfa = {self.mma_tiler_sfa}")
        print(f"[tiler] mma_tiler_sfb = {self.mma_tiler_sfb}")
        # Number of SFA LDGSTS.32 per K tile: each loads 4 SF blocks (4 bytes)
        self.num_sfa_ldgsts = mma_inst_tile_k  # = K_gemm_tile / (sf_vec_size * 4)
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        # CTA-level SFB tile shape (used for SFB TMEM column accounting in
        # overlapping-accum). N-mode rounds MMA_N up to 128 like SFB MMA.
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        self.epi_tile = utils.sm100.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.d_layout,
            self.d_dtype,
        )
        # N-extent of one epilogue subtile (used by overlapping-accum early release).
        self.epi_tile_n = cute.size(self.epi_tile[1])

        self.smem_capacity = utils.get_smem_capacity_in_bytes()

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_d_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.d_dtype,
            self.d_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
        )

        # Overlapping-accum: when only one accumulator stage fits in TMEM
        # (MMA_N == 256), squeeze a second logical acc buffer into the columns
        # otherwise reserved for SFA/SFB so the math mainloop of the next tile
        # can overlap the epilogue of the current tile.
        self.overlapping_accum = self.num_acc_stage == 1
        self.num_sfa_tmem_cols = (
            self.cta_tile_shape_mnk[0] // 32
        ) * self.mma_inst_tile_k
        self.num_sfb_tmem_cols = (
            self.cta_tile_shape_mnk_sfb[1] // 32
        ) * self.mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        # Subtile index (in reverse-subtile order) at which the shared overlap
        # region becomes free and the accumulator can be released early.
        self.iter_acc_early_release_in_epilogue = (
            self.num_sf_tmem_cols // self.epi_tile_n
        )

        # Setup CLC stage (single-stage CLC pipeline)
        self.num_clc_stage = 1
        assert self.num_clc_stage == 1, "Only single-stage CLC pipeline is supported"

        # Compute A/B/C shared memory layout
        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        # Flatten hierarchical K for blockscaled utils
        # Conv uses (K,) tuple for structured K dim, but blockscaled utils expect scalar K
        flat_k = (
            self.mma_tiler[2]
            if isinstance(self.mma_tiler[2], int)
            else self.mma_tiler[2][0]
        )
        flat_mma_tiler = (self.mma_tiler[0], self.mma_tiler[1], flat_k)
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            flat_mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            flat_mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )

        self.d_smem_layout_staged = utils.sm100.make_smem_layout_epi(
            self.d_dtype, self.d_layout, self.epi_tile, self.num_d_stage
        )

        # Compute the number of tensor memory allocation columns.
        # For overlapping-accum the second acc buffer is strided into the
        # SFA/SFB columns, so the inherited helper (which builds a contiguous
        # 2-stage fake) would under-count. Derive the column count from the
        # same fake tensor used at the injection sites instead.
        if cutlass.const_expr(self.overlapping_accum):
            self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(
                self._make_acc_fake_tensor(tiled_mma, self.mma_tiler),
                arch=self.arch,
            )
        else:
            self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
                tiled_mma, self.mma_tiler, self.num_acc_stage, self.arch
            )

        # Compute preferred cluster layout for dual-cluster scheduling
        self.preferred_cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.preferred_cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.num_preferred_mcast_ctas_a = cute.size(
            self.preferred_cluster_layout_vmnk.shape[2]
        )
        self.num_preferred_mcast_ctas_b = cute.size(
            self.preferred_cluster_layout_vmnk.shape[1]
        )
        self.is_preferred_a_mcast = self.num_preferred_mcast_ctas_a > 1
        self.is_preferred_b_mcast = self.num_preferred_mcast_ctas_b > 1

        # Fallback cluster layout was already computed as cluster_layout_vmnk above
        self.fallback_cluster_layout_vmnk = self.cluster_layout_vmnk
        self.num_fallback_mcast_ctas_a = self.num_mcast_ctas_a
        self.num_fallback_mcast_ctas_b = self.num_mcast_ctas_b
        self.is_fallback_a_mcast = self.is_a_mcast
        self.is_fallback_b_mcast = self.is_b_mcast

        # SFB cluster layouts (SFB always uses 1CTA group, thr_id shape = 1)
        self.preferred_cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.preferred_cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )
        self.fallback_cluster_layout_sfb_vmnk = self.cluster_layout_sfb_vmnk

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        d: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        alpha: cute.Tensor,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        sfd_tensor: Optional[cute.Tensor] = None,
        norm_const_tensor: Optional[cute.Tensor] = None,
        # Conv pad/stride/dilation as cute.compile entry scalars. The caller
        # passes boxed cutlass.Int32(...) so they lower to runtime SSA: one
        # cubin then serves any pad/stride/dil because the SAME scalars feed
        # BOTH the im2col A descriptor corner (host-side, in _setup_conv_tma)
        # AND the device per-tile coord math. Boxing matters: a raw Python int
        # here would embed its value in the mangled function name, defeating
        # reuse and forcing recompilation per config.
        rt_upper_pad_d: cutlass.Int32 = None,
        rt_upper_pad_h: cutlass.Int32 = None,
        rt_upper_pad_w: cutlass.Int32 = None,
        rt_lower_pad_d: cutlass.Int32 = None,
        rt_lower_pad_h: cutlass.Int32 = None,
        rt_lower_pad_w: cutlass.Int32 = None,
        rt_stride_d: cutlass.Int32 = None,
        rt_stride_h: cutlass.Int32 = None,
        rt_stride_w: cutlass.Int32 = None,
        rt_dil_d: cutlass.Int32 = None,
        rt_dil_h: cutlass.Int32 = None,
        rt_dil_w: cutlass.Int32 = None,
    ):
        """Execute the persistent convolution operation with dynamic preferred cluster scheduling.

        :param a: Input tensor A - (N, D, H, W, C) layout
        :param b: Filter tensor B - (K, T, R, S, C) layout
        :param d: Output tensor D - (N, Z, P, Q, K) layout
        :param sfa_tensor: Block scale factor tensor for A
        :param sfb_tensor: Block scale factor tensor for B
        :param alpha: 1-element FP32 device tensor; applied to the accumulator
            in FP32 before output quantization (matches TRT-LLM GEMM convention)
        :param stream: CUDA stream for asynchronous execution
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :param sfd_tensor: Output scale factor tensor (NVFP4 SFD); pass None to skip SFD generation
        :param norm_const_tensor: 1-element FP32 device tensor; per-tensor amax-derived
            global FP4 scale, multiplied into the SFD encode step. Required when
            sfd_tensor is provided.
        """
        self._setup_conv_input_attrs(a, b, d)
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        # SFD shares the input sf_dtype (NVFP4 standard: Float8E4M3FN, vec=16).
        self.sfd_dtype: Type[cutlass.Numeric] = self.sf_dtype
        # SFD opt-in: caller supplies BOTH sfd_tensor and norm_const_tensor.
        # SFD is only valid with FP4 output (matches TRT-LLM GEMM SFC path).
        self.gen_sfd: bool = sfd_tensor is not None and norm_const_tensor is not None
        if cutlass.const_expr(
            self.gen_sfd and self.d_dtype is not cutlass.Float4E2M1FN
        ):
            raise ValueError(
                "SFD is only supported for Float4E2M1FN (NVFP4) output; "
                f"got d_dtype={self.d_dtype}"
            )
        self._setup_attributes()

        # Pad/stride/dilation operands that BOTH the im2col A descriptor corner
        # and the device per-tile coord math consume. They are the runtime
        # Int32 scalars passed at the cute.compile entry, so one cubin serves
        # any config. Threading the SAME values into the descriptor and the
        # kernel is required: runtime-ing only one side would let A's load
        # coords (descriptor) desync from SFA's cp.async coords (device),
        # zeroing the accumulator on any non-default config.
        upper_pad_op = (rt_upper_pad_d, rt_upper_pad_h, rt_upper_pad_w)
        lower_pad_op = (rt_lower_pad_d, rt_lower_pad_h, rt_lower_pad_w)
        stride_op = (rt_stride_d, rt_stride_h, rt_stride_w)
        dil_op = (rt_dil_d, rt_dil_h, rt_dil_w)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_tiler[:2],
        )

        (
            tma_atom_a_preferred,
            tma_tensor_a_preferred,
            tma_atom_a_fallback,
            tma_tensor_a_fallback,
            tma_atom_b_preferred,
            tma_tensor_b_preferred,
            tma_atom_b_fallback,
            tma_tensor_b_fallback,
            tma_atom_d,
            tma_tensor_d,
        ) = self._setup_conv_tma(
            a, b, d, tiled_mma, upper_pad_op, lower_pad_op, stride_op, dil_op
        )

        # SFB tiled_mma (always 1CTA group for SFB)
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # SFB tensor reshape: match B's GEMM view (K, C*S*R*T) for correct TMA tiling.
        # B filter is (K, T, R, S, C) -> reorder to (K, C, S, R, T) -> group to (K, C*S*R*T).
        # This makes SFB N dim = K (GEMM N) and SFB K dim = C*S*R*T/16 (GEMM K / sf_vec).
        # Use a flat integer K extent for tile_atom_to_shape_SF, NOT the hierarchical
        # shape from group_modes: group_modes produces (K, (C, S, R, T)) with nested
        # strides, which tile_atom_to_shape_SF interprets incorrectly vs the flat
        # swizzled SFB storage.
        #
        # The N extent MUST be the dynamic b.shape[0], not a trace-time-static int.
        # For cta_n==192 the SFB layout is reshaped into overlapping 256-wide windows
        # ((2,2),y) with y=ceil_div(N_sf,4) below. A static N collapses y to a literal
        # 1, and CuTe coalesces the size-1 sub-mode away, flattening the window RestN
        # from (2,?) to 2. The scheduler still emits ceil(N/192) cta-tiles, so for a
        # partial last tile slice_n runs OOB on the flattened RestN and the TMA load
        # clamps to zeros -> wrong (zero) SFB scaling on that tile. A dynamic N keeps y
        # symbolic, blocks the coalesce, and slice_n decomposes to an in-bounds coord.
        # input_C stays a host int because it shapes the mma_tiler/SMEM.
        b_shape_for_sfb = (
            b.shape[0],
            self.input_C * b.shape[1] * b.shape[2] * b.shape[3],
            1,
        )
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_shape_for_sfb, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

        # SFB TMA setup (dual atoms for preferred + fallback cluster shapes)
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )

        def _make_sfb_tma(cluster_shape_mn, cluster_layout_sfb_vmnk):
            sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
                cluster_shape_mn, tiled_mma.thr_id
            )
            atom, tensor = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                sfb_tensor,
                sfb_smem_layout,
                self.mma_tiler_sfb,
                tiled_mma_sfb,
                cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )
            # Special-case for N=192: align the SFB scale-factor layout for Tensor Memory packing
            if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                x = tensor.stride[0][1]
                y = cute.ceil_div(tensor.shape[0][1], 4)
                new_shape = (
                    (tensor.shape[0][0], ((2, 2), y)),
                    tensor.shape[1],
                    tensor.shape[2],
                )
                x_times_3 = 3 * x
                new_stride = (
                    (tensor.stride[0][0], ((x, x), x_times_3)),
                    tensor.stride[1],
                    tensor.stride[2],
                )
                tensor = cute.make_tensor(
                    tensor.iterator,
                    cute.make_layout(new_shape, stride=new_stride),
                )
            return atom, tensor

        tma_atom_sfb_preferred, tma_tensor_sfb_preferred = _make_sfb_tma(
            self.preferred_cluster_shape_mn, self.preferred_cluster_layout_sfb_vmnk
        )
        tma_atom_sfb_fallback, tma_tensor_sfb_fallback = _make_sfb_tma(
            self.fallback_cluster_shape_mn, self.fallback_cluster_layout_sfb_vmnk
        )

        # Compute A, B, SFB copy sizes for separate TMA pipelines.
        # SFB has its own pipeline because B and SFB use different mcast
        # masks (cluster_layout_vmnk vs cluster_layout_sfb_vmnk), so bundling
        # them into one pipeline can stall TX accumulation in cluster > 1.
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_a_bytes = a_copy_size * atom_thr_size
        self.num_tma_load_b_bytes = b_copy_size * atom_thr_size
        self.num_tma_load_sfb_bytes = sfb_copy_size * atom_thr_size

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            # Shared TMA pipeline barriers for A, B and SFB
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            # CPASYNC SFA pipeline barriers
            sfa_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # CLC pipeline barriers and response buffer
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_clc_stage * 2]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int32, self.num_clc_response_bytes // 4 * self.num_clc_stage
                ],
                16,
            ]
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sD: cute.struct.Align[
                cute.struct.MemRange[
                    self.d_dtype,
                    cute.cosize(self.d_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Compute grid size and scheduler params for both cluster shapes
        self.fallback_tile_sched_params, _ = self._compute_grid(
            tma_tensor_d,
            self.cta_tile_shape_mnk,
            self.fallback_cluster_shape_mn,
            self.swizzle_size,
            self.raster_along,
        )
        self.preferred_tile_sched_params, preferred_grid = self._compute_grid(
            tma_tensor_d,
            self.cta_tile_shape_mnk,
            self.preferred_cluster_shape_mn,
            self.swizzle_size,
            self.raster_along,
        )
        # Extract conv spatial dims from input/output tensors
        # a: (N, D, H, W, C), d: (N, Z, P, Q, K)
        conv_N = cute.size(a, mode=[0])
        conv_D = cute.size(a, mode=[1])
        conv_H = cute.size(a, mode=[2])
        conv_W = cute.size(a, mode=[3])
        conv_Z = cute.size(d, mode=[1])
        conv_P = cute.size(d, mode=[2])
        conv_Q = cute.size(d, mode=[3])
        # Filter T/R/S from the dynamic-layout filter tensor b (K, T, R, S, C):
        # runtime Int32 so one cubin serves any T/R/S. Fed to the device
        # kernel's conv_T/R/S params.
        conv_T, conv_R, conv_S = b.shape[1], b.shape[2], b.shape[3]
        # stride/dil/pad come from the SAME operand tuples that built the A
        # descriptor corner above, so the device's SFA cp.async coords stay in
        # lockstep with A's im2col load coords under any runtime config.
        stride_d, stride_h, stride_w = stride_op
        dil_d, dil_h, dil_w = dil_op
        # SFA cp.async path uses the leading (lower) pad to mirror what the
        # im2col TMA A descriptor subtracts when mapping output→input coords:
        #   d_in = z*stride - lower_pad + t*dil
        # Using upper_padding here desyncs SFA from A on asymmetric pads,
        # making A's nonzero positions multiply SFA's zero positions → acc=0.
        pad_d, pad_h, pad_w = lower_pad_op
        K_gemm_tile = self.mma_tiler[2][0]  # mma_inst_shape_k * mma_inst_tile_k

        # Build mSFD tensor: same logical shape as mD (((Q,P,Z,N), K, 1)) but
        # with the K dimension expressed as (sfd_vec_size, sf_k_padded) with
        # strides (0, 1) so that sfd_vec_size consecutive K elements share one
        # physical scale factor. Storage layout matches SFA's K-contig-in-M
        # form: (mn=NZPQ, sf_k_padded, 1) with mn-stride = sf_k_padded.
        if cutlass.const_expr(self.gen_sfd):
            # The SFD store is a SIMT autovec_copy (no TMA), so this global
            # descriptor's shape feeds a register fragment that must be
            # statically shaped. Source N/Z/P/Q/K from host ints (not the
            # dynamic-tensor cute.size values) so the layout stays trace-const.
            sfd_N, sfd_Z, sfd_P, sfd_Q = self.output_nzpq
            conv_K = self.output_K
            sf_k = (conv_K + self.sfd_vec_size - 1) // self.sfd_vec_size
            sf_k_padded = ((sf_k + 3) // 4) * 4  # pad sf_k to multiple of 4
            stride_mn = sf_k_padded
            mSFD_layout = cute.make_layout(
                (
                    (sfd_Q, sfd_P, sfd_Z, sfd_N),
                    (self.sfd_vec_size, sf_k_padded),
                    1,
                ),
                stride=(
                    (
                        stride_mn,
                        sfd_Q * stride_mn,
                        sfd_P * sfd_Q * stride_mn,
                        sfd_Z * sfd_P * sfd_Q * stride_mn,
                    ),
                    (0, 1),
                    sfd_N * sfd_Z * sfd_P * sfd_Q * stride_mn,
                ),
            )
            mSFD_mnl = cute.make_tensor(sfd_tensor.iterator, mSFD_layout)
        else:
            mSFD_mnl = None

        # Launch the megakernel synchronously (dispatcher selects preferred vs fallback)
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a_preferred,
            tma_tensor_a_preferred,
            tma_atom_a_fallback,
            tma_tensor_a_fallback,
            tma_atom_b_preferred,
            tma_tensor_b_preferred,
            tma_atom_b_fallback,
            tma_tensor_b_fallback,
            sfa_tensor,
            tma_atom_sfb_preferred,
            tma_tensor_sfb_preferred,
            tma_atom_sfb_fallback,
            tma_tensor_sfb_fallback,
            tma_atom_d,
            tma_tensor_d,
            mSFD_mnl,
            alpha,
            norm_const_tensor,
            self.preferred_cluster_layout_vmnk,
            self.fallback_cluster_layout_vmnk,
            self.preferred_cluster_layout_sfb_vmnk,
            self.fallback_cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.d_smem_layout_staged,
            self.epi_tile,
            self.preferred_tile_sched_params,
            self.fallback_tile_sched_params,
            epilogue_op,
            conv_T,
            conv_R,
            conv_S,
            stride_d,
            stride_h,
            stride_w,
            dil_d,
            dil_h,
            dil_w,
            pad_d,
            pad_h,
            pad_w,
            conv_D,
            conv_H,
            conv_W,
            conv_Z,
            conv_P,
            conv_Q,
            conv_N,
            self.input_C,
            K_gemm_tile,
        ).launch(
            grid=preferred_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.preferred_cluster_shape_mn, 1),
            fallback_cluster=(*self.fallback_cluster_shape_mn, 1),
            stream=stream,
            smem_merge_branch_allocs=True,
        )
        return

    # GPU device kernel - megakernel dispatcher for preferred/fallback cluster
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a_preferred: cute.CopyAtom,
        mA_mkl_preferred: cute.Tensor,
        tma_atom_a_fallback: cute.CopyAtom,
        mA_mkl_fallback: cute.Tensor,
        tma_atom_b_preferred: cute.CopyAtom,
        mB_nkl_preferred: cute.Tensor,
        tma_atom_b_fallback: cute.CopyAtom,
        mB_nkl_fallback: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb_preferred: cute.CopyAtom,
        mSFB_nkl_preferred: cute.Tensor,
        tma_atom_sfb_fallback: cute.CopyAtom,
        mSFB_nkl_fallback: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSFD_mnl: Optional[cute.Tensor],
        alpha: cute.Tensor,
        norm_const_tensor: Optional[cute.Tensor],
        preferred_cluster_layout_vmnk: cute.Layout,
        fallback_cluster_layout_vmnk: cute.Layout,
        preferred_cluster_layout_sfb_vmnk: cute.Layout,
        fallback_cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        d_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        preferred_tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        fallback_tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        # Conv parameters for SFA im2col coordinate mapping. conv_T/R/S are
        # runtime Int32 so one cubin serves any filter size; they only feed
        # register comparisons in the K-loop (S->R->T wrap), adding no IDIV.
        conv_T: cutlass.Int32,
        conv_R: cutlass.Int32,
        conv_S: cutlass.Int32,
        # pad/stride/dilation are runtime Int32 so one cubin serves any conv
        # config; they feed the per-tile base-coord arithmetic (z*stride - pad)
        # and the K-loop dilation step, swapping immediates for registers with
        # no new IDIV.
        stride_d: cutlass.Int32,
        stride_h: cutlass.Int32,
        stride_w: cutlass.Int32,
        dil_d: cutlass.Int32,
        dil_h: cutlass.Int32,
        dil_w: cutlass.Int32,
        pad_d: cutlass.Int32,
        pad_h: cutlass.Int32,
        pad_w: cutlass.Int32,
        # Coordinate geometry: runtime Int32 so the per-tile NDHW decomposition
        # (m_global -> n,z,p,q) lowers to real IDIV and one cubin serves any
        # N/D/H/W/Z/P/Q.
        input_D: cutlass.Int32,
        input_H: cutlass.Int32,
        input_W: cutlass.Int32,
        output_Z: cutlass.Int32,
        output_P: cutlass.Int32,
        output_Q: cutlass.Int32,
        input_N: cutlass.Int32,
        input_C: cutlass.Constexpr,
        K_gemm_tile: cutlass.Constexpr,
    ):
        """
        GPU device kernel entry point: dispatches to preferred or fallback cluster path.
        """
        # Determine if this CTA is launched in a preferred-shape cluster
        cbdim_x, cbdim_y, cbdim_z = cute.arch.block_in_cluster_dim()
        is_preferred_cluster = (
            cbdim_x == self.preferred_cluster_shape_mn[0]
            and cbdim_y == self.preferred_cluster_shape_mn[1]
            and cbdim_z == 1
        )

        # Megakernel: only one branch executes per launch.
        # smem_merge_branch_allocs=True at launch enables shared memory reuse between two paths.
        if is_preferred_cluster:
            self.cluster_specific_kernel(
                tiled_mma,
                tiled_mma_sfb,
                tma_atom_a_preferred,
                mA_mkl_preferred,
                tma_atom_b_preferred,
                mB_nkl_preferred,
                mSFA_mkl,
                tma_atom_sfb_preferred,
                mSFB_nkl_preferred,
                tma_atom_d,
                mC_mnl,
                mSFD_mnl,
                alpha,
                norm_const_tensor,
                preferred_cluster_layout_vmnk,
                preferred_cluster_layout_sfb_vmnk,
                a_smem_layout_staged,
                b_smem_layout_staged,
                sfa_smem_layout_staged,
                sfb_smem_layout_staged,
                d_smem_layout_staged,
                epi_tile,
                preferred_tile_sched_params,
                epilogue_op,
                self.num_preferred_mcast_ctas_a + self.num_preferred_mcast_ctas_b - 1,
                self.is_preferred_a_mcast,
                self.is_preferred_b_mcast,
                self.preferred_cluster_shape_mn,
                conv_T,
                conv_R,
                conv_S,
                stride_d,
                stride_h,
                stride_w,
                dil_d,
                dil_h,
                dil_w,
                pad_d,
                pad_h,
                pad_w,
                input_D,
                input_H,
                input_W,
                output_Z,
                output_P,
                output_Q,
                input_N,
                input_C,
                K_gemm_tile,
            )
        else:
            self.cluster_specific_kernel(
                tiled_mma,
                tiled_mma_sfb,
                tma_atom_a_fallback,
                mA_mkl_fallback,
                tma_atom_b_fallback,
                mB_nkl_fallback,
                mSFA_mkl,
                tma_atom_sfb_fallback,
                mSFB_nkl_fallback,
                tma_atom_d,
                mC_mnl,
                mSFD_mnl,
                alpha,
                norm_const_tensor,
                fallback_cluster_layout_vmnk,
                fallback_cluster_layout_sfb_vmnk,
                a_smem_layout_staged,
                b_smem_layout_staged,
                sfa_smem_layout_staged,
                sfb_smem_layout_staged,
                d_smem_layout_staged,
                epi_tile,
                fallback_tile_sched_params,
                epilogue_op,
                self.num_fallback_mcast_ctas_a + self.num_fallback_mcast_ctas_b - 1,
                self.is_fallback_a_mcast,
                self.is_fallback_b_mcast,
                self.fallback_cluster_shape_mn,
                conv_T,
                conv_R,
                conv_S,
                stride_d,
                stride_h,
                stride_w,
                dil_d,
                dil_h,
                dil_w,
                pad_d,
                pad_h,
                pad_w,
                input_D,
                input_H,
                input_W,
                output_Z,
                output_P,
                output_Q,
                input_N,
                input_C,
                K_gemm_tile,
            )

    @cute.jit()
    def cluster_specific_kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSFD_mnl: Optional[cute.Tensor],
        alpha: cute.Tensor,
        norm_const_tensor: Optional[cute.Tensor],
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        d_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        num_tma_producer: int,
        effective_is_a_mcast: bool,
        effective_is_b_mcast: bool,
        cluster_shape: Tuple[int, int],
        # Conv parameters for SFA im2col coordinate mapping. conv_T/R/S are
        # runtime Int32 so one cubin serves any filter size; they only feed
        # register comparisons in the K-loop (S->R->T wrap), adding no IDIV.
        conv_T: cutlass.Int32,
        conv_R: cutlass.Int32,
        conv_S: cutlass.Int32,
        # pad/stride/dilation are runtime Int32 so one cubin serves any conv
        # config; they feed the per-tile base-coord arithmetic (z*stride - pad)
        # and the K-loop dilation step, swapping immediates for registers with
        # no new IDIV.
        stride_d: cutlass.Int32,
        stride_h: cutlass.Int32,
        stride_w: cutlass.Int32,
        dil_d: cutlass.Int32,
        dil_h: cutlass.Int32,
        dil_w: cutlass.Int32,
        pad_d: cutlass.Int32,
        pad_h: cutlass.Int32,
        pad_w: cutlass.Int32,
        # Coordinate geometry: runtime Int32 so the per-tile NDHW decomposition
        # (m_global -> n,z,p,q) lowers to real IDIV and one cubin serves any
        # N/D/H/W/Z/P/Q.
        input_D: cutlass.Int32,
        input_H: cutlass.Int32,
        input_W: cutlass.Int32,
        output_Z: cutlass.Int32,
        output_P: cutlass.Int32,
        output_Q: cutlass.Int32,
        input_N: cutlass.Int32,
        input_C: cutlass.Constexpr,
        K_gemm_tile: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the CLC dynamic persistent convolution computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            # cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_d)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        is_first_cta_in_cluster = cta_rank_in_cluster == 0
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Pipeline Init: merged pipeline shares one mbarrier between TMA (A+B+SFB)
        # and cp.async (SFA) producers, for any cluster shape.
        # tx_count covers A+B+SFB TMA bytes; arrive_count = 128 cp.async lanes +
        # 1 TMA arrive. The TMA arrive is produced by the leader's multicast
        # arrive_and_expect_tx and propagated to every peer barrier by hardware,
        # so peer CTAs must not issue an additional arrive.
        # For 2CTA (use_2cta_instrs), peer CTA's 128 cpasync threads also
        # remote-arrive on leader's sfa_full via DSMEM (shift-by-K pattern)
        # to eliminate the peer-sSFA race where leader MMA proceeded before
        # peer's LDGSTS landed. Leader's arrive_count must therefore be
        # bumped to 129 + 128 = 257.
        # num_tma_producer is provided as kernel parameter (per-cluster)
        sfa_cpasync_arrive_count = 257 if use_2cta_instrs else 129
        cpasynd_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, sfa_cpasync_arrive_count
        )
        merged_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        merged_tx_count = (
            self.num_tma_load_a_bytes
            + self.num_tma_load_b_bytes
            + self.num_tma_load_sfb_bytes
        )
        ab_sfa_pipeline = PipelineTmaCpAsyncUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            cpasynd_producer_group=cpasynd_producer_group,
            consumer_group=merged_consumer_group,
            tx_count=merged_tx_count,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Pipeline Init: Initialize acd_pipeline (barrier) and states
        acd_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acd_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acd_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acd_pipeline_producer_group,
            consumer_group=acd_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Pipeline Init: Initialize clc_pipeline (CLC fetch async)
        # Consumers of CLC response: TMA(1) + LDGSTS_SFA(4) + MMA(1) + Epilogue(4) per CTA, * cluster_size
        # The tile-coord query is broadcast to all CTAs in the cluster.
        clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(cluster_shape)
        num_clc_consumer_threads = 32 * (
            1
            + cluster_size
            * (1 + len(self.ldgsts_sfa_warp_id) + len(self.epilogue_warp_id) + 1)
        )
        clc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_clc_consumer_threads
        )
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbar_ptr.data_ptr(),
            num_stages=self.num_clc_stage,
            producer_group=clc_pipeline_producer_group,
            consumer_group=clc_pipeline_consumer_group,
            tx_count=self.num_clc_response_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Pipeline Init: Tensor memory alloc/dealloc barrier init
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem_dealloc_barrier = None
        if cutlass.const_expr(not self.use_tma_store):
            tmem_dealloc_barrier = pipeline.NamedBarrier(
                barrier_id=self.tmem_dealloc_sync_bar_id,
                num_threads=32 * len(self.epilogue_warp_id),
            )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_shape, is_relaxed=True)

        # CLC response buffer pointer + consumer state
        clc_response_ptr = storage.clc_response.data_ptr()
        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_clc_stage
        )

        #
        # Setup smem tensor A/B/SFA/SFB/D
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sD = storage.sD.get_tensor(
            d_smem_layout_staged.outer, swizzle=d_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfb_full_mcast_mask = None

        if cutlass.const_expr(
            effective_is_a_mcast or effective_is_b_mcast or use_2cta_instrs
        ):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )

        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            cute.slice_(self.mma_tiler_sfa, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        gD_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL) - same MNL tiling as gD; the K-axis (N
        # of MMA) carries a stride-0 broadcast over self.sfd_vec_size groups.
        if cutlass.const_expr(self.gen_sfd):
            gSFD_mnl = cute.local_tile(
                mSFD_mnl,
                cute.slice_(self.mma_tiler, (None, None, 0)),
                (None, None, None),
            )
        else:
            gSFD_mnl = None

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgD = thr_mma.partition_C(gD_mnl)
        # SFD: same partition shape as gD; broadcast strides preserved through partition.
        if cutlass.const_expr(self.gen_sfd):
            tCgSFD = thr_mma.partition_C(gSFD_mnl)
        else:
            tCgSFD = None

        #
        # Partition global/shared tensor for TMA load B/SFB
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N, STAGE)
        # For overlapping-accum this carries the strided 2-buffer layout so the
        # accumulator's second stage aliases the SFA/SFB columns.
        tCtAcc_fake = self._make_acc_fake_tensor(tiled_mma, self.mma_tiler)

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=cluster_shape)

        #
        # Specialized TMA load A/B/SFA/SFB warp
        #
        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Special-case for N=64: SFB scale-factor slicing to match Tensor Memory packing
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2
                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                # Peek (try_wait) shared AB/SFB buffer empty
                ab_producer_state.reset_count()
                peek_ab_empty_status = ab_sfa_pipeline.producer_try_acquire(
                    ab_producer_state
                )

                #
                # Tma load loop with increment_coord for im2col K traversal
                # Traversal order: S innermost → R → T → C outermost
                #
                # A's RestK shape from im2col TMA is (C_tiles, S, R, T) where
                # C_tiles is leftmost. To traverse S→R→T→C, we build a
                # permuted traversal shape and remap coords when indexing.
                k_shape = cute.shape(tAgA_slice, mode=1)  # (C_tiles, S, R, T)
                k_C_tiles = cute.size(k_shape, mode=0)
                k_S = cute.size(k_shape, mode=1)
                k_R = cute.size(k_shape, mode=2)
                k_T = cute.size(k_shape, mode=3)
                # Traversal shape in S→R→T→C order (colexicographic on this)
                trav_shape = (k_S, k_R, k_T, k_C_tiles)
                trav_coord = cute.repeat_like(0, trav_shape)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # TMA producer path on the merged pipeline (leader does
                    # arrive_and_expect_tx, peer does nothing).
                    ab_sfa_pipeline.producer_acquire_tma(
                        ab_producer_state, peek_ab_empty_status
                    )

                    # Remap (s,r,t,c) → (c,s,r,t) to match A/B RestK layout
                    ab_coord = (
                        trav_coord[3],  # C_tiles
                        trav_coord[0],  # S
                        trav_coord[1],  # R
                        trav_coord[2],  # T
                    )

                    # TMA load A (use multi-dim coord for im2col K indexing)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_coord)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_sfa_pipeline.producer_get_barrier(
                            ab_producer_state
                        ),
                        mcast_mask=a_full_mcast_mask,
                    )
                    # TMA load B (multi-dim K layout, use same coord)
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_coord)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_sfa_pipeline.producer_get_barrier(
                            ab_producer_state
                        ),
                        mcast_mask=b_full_mcast_mask,
                    )
                    # TMA load SFB: compute flat K tile index from (c,s,r,t)
                    # SFB uses flat layout with C→S→R→T physical order
                    sfb_flat_k = (
                        ab_coord[0]
                        + ab_coord[1] * k_C_tiles
                        + ab_coord[2] * k_S * k_C_tiles
                        + ab_coord[3] * k_R * k_S * k_C_tiles
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, sfb_flat_k)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_sfa_pipeline.producer_get_barrier(
                            ab_producer_state
                        ),
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Peek (try_wait) shared AB/SFB buffer empty for next iteration
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_sfa_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                    # Advance traversal coord in S→R→T→C order
                    trav_coord = cute.increment_coord(trav_coord, trav_shape)

                #
                # Advance to next tile (CLC consumer)
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            #
            # Wait shared AB/SFB buffer empty
            #
            ab_sfa_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized LDGSTS SFA warp
        #
        if (
            warp_idx >= self.ldgsts_sfa_warp_id[0]
            and warp_idx <= self.ldgsts_sfa_warp_id[-1]
        ):
            #
            # Setup SFA CPASYNC copy atom
            #
            sfa_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mSFA_mkl.element_type,
                num_bits_per_copy=32,
            )
            tidx_in_warpgroup = tidx % 128

            # SFA predicate: dynamically computed per LDGSTS for conv3x3
            sfa_predicate_tensor = cute.make_rmem_tensor(
                cute.make_layout((1,)),
                cutlass.Boolean,
            )

            # Identity M offset (no permutation) + CTA offset within cluster
            cta_m_offset = mma_tile_coord_v * self.cta_tile_shape_mnk[0]
            sfa_m_offset = (
                cta_m_offset
                + 8 * (tidx_in_warpgroup // 32)
                + 32 * ((tidx_in_warpgroup % 32) // 8)
                + (tidx_in_warpgroup % 8)
            )

            # Slice sSFA for this thread's M-row and sub-row
            tAsSFA = sSFA[
                (
                    (
                        (
                            (
                                8 * (tidx_in_warpgroup // 32) + (tidx_in_warpgroup % 8),
                                (tidx_in_warpgroup % 32) // 8,
                            ),
                            None,
                        ),
                        None,
                    ),
                    None,
                    None,
                    None,
                )
            ]

            # SFA global tensor: shape (MN, SF_K, L) with K contiguous
            # For conv3x3 with arbitrary C, each LDGSTS may load from a different
            # input element (different M address) depending on the filter position.
            # sf_k_per_fpos = C // sf_vec_size: SF K blocks per filter position
            # When C < K_gemm_tile, a single K tile spans multiple filter positions.

            # Precompute conv spatial constants
            K_gemm_tile = self.mma_tiler[2][0]
            ZPQ = output_Z * output_P * output_Q
            PQ = output_P * output_Q
            DHW = input_D * input_H * input_W
            HW = input_H * input_W
            CTA_M = self.cta_tile_shape_mnk[0]
            sf_k_per_fpos = input_C // self.sf_vec_size
            # SF blocks loaded per k_tile (may be less than sf_k_per_fpos when
            # C > K_gemm_tile, i.e. one fpos spans multiple k_tiles)
            sf_k_per_ktile = K_gemm_tile // self.sf_vec_size
            # Number of C tiles per filter position
            c_tiles_per_fpos = input_C // K_gemm_tile

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            work_tile = tile_sched.initial_work_tile_info()

            sfa_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Compute this thread's global M index (output pixel)
                # Use mma_tile_coord_mnl[0] (M tile index without V) * full MMA tile M
                # because sfa_m_offset already includes the V (CTA-within-pair) offset
                cta_m_tile_offset = (
                    mma_tile_coord_mnl[0] * CTA_M * cute.size(tiled_mma.thr_id.shape)
                )
                m_global = cta_m_tile_offset + sfa_m_offset

                # Decompose m_global -> (n, z, p, q) output coordinates
                n_idx = m_global // ZPQ
                zpq_rem = m_global % ZPQ
                z_idx = zpq_rem // PQ
                pq_rem = zpq_rem % PQ
                p_idx = pq_rem // output_Q
                q_idx = pq_rem % output_Q

                # Check if m_global is within valid output range
                m_valid = m_global < (input_N * ZPQ)

                # Initialize base spatial coordinates for filter pos (t=0, r=0, s=0)
                d_in_base = z_idx * stride_d - pad_d
                h_in_base = p_idx * stride_h - pad_h
                w_in_base = q_idx * stride_w - pad_w
                n_clamped = n_idx if m_valid else 0

                # Maintain (s,r,t,c) coords for S→R→T→C traversal order
                # matching TMA warp's coord_iter order
                sfa_s_idx = 0
                sfa_r_idx = 0
                sfa_t_idx = 0
                sfa_c_tile_idx = 0

                # Peek (try_wait) SFA buffer empty
                sfa_producer_state.reset_count()
                peek_sfa_empty_status = cutlass.Boolean(1)
                if sfa_producer_state.count < k_tile_cnt:
                    peek_sfa_empty_status = ab_sfa_pipeline.producer_try_acquire(
                        sfa_producer_state
                    )

                #
                # Shift-by-K cross-CTA commit ringbuffer (K=2). Per iter,
                # commit the LDGSTS as a cp.async group and wait_group(K);
                # then remote-arrive on leader's sfa_full for the stage from
                # K iters ago. Guarantees that LDGSTS has landed globally
                # before leader MMA sees the barrier full.
                #
                sfa_shift_K = 2
                pending_a = cutlass.Int32(0)
                pending_b = cutlass.Int32(0)

                #
                # CPASYNC SFA load loop with coordinate-based im2col
                # Traversal order: S→R→T→C (matching TMA warp)
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for SFA buffer empty
                    ab_sfa_pipeline.producer_acquire(
                        sfa_producer_state, peek_sfa_empty_status
                    )

                    cur_stage = sfa_producer_state.index

                    tAsSFA_ktile = tAsSFA[(None, None, None, None, cur_stage)]

                    # Compute unclamped spatial coords from current (t,r,s)
                    d_in = d_in_base + sfa_t_idx * dil_d
                    h_in = h_in_base + sfa_r_idx * dil_h
                    w_in = w_in_base + sfa_s_idx * dil_w

                    # Clamp for safe gmem address
                    d_cl = d_in if d_in >= 0 else 0
                    d_cl = d_cl if d_cl < input_D else 0
                    h_cl = h_in if h_in >= 0 else 0
                    h_cl = h_cl if h_cl < input_H else 0
                    w_cl = w_in if w_in >= 0 else 0
                    w_cl = w_cl if w_cl < input_W else 0
                    sfa_m_addr = n_clamped * DHW + d_cl * HW + h_cl * input_W + w_cl

                    # Predicate: m_valid AND spatial bounds check
                    sfa_pred_val = cutlass.Boolean(0)
                    if m_valid:
                        if d_in >= 0 and d_in < input_D:
                            if h_in >= 0 and h_in < input_H:
                                if w_in >= 0 and w_in < input_W:
                                    sfa_pred_val = cutlass.Boolean(1)

                    # SF K base offset for current C tile within this fpos
                    sf_k_base = sfa_c_tile_idx * sf_k_per_ktile

                    # LDGSTS.32 SFA — all SI share same fpos
                    for i in range(self.num_sfa_ldgsts):
                        SI = ((tidx_in_warpgroup % 32) // 8 ^ i) % self.num_sfa_ldgsts

                        sf_k_smem = SI * 4
                        local_sf_k = sf_k_base + (sf_k_smem % sf_k_per_ktile)

                        sfa_predicate_tensor[0] = sfa_pred_val

                        # Gmem: load from this fpos's M address at local_sf_k offset
                        sfa_m_slice = mSFA_mkl[(sfa_m_addr, None, 0)]
                        tAgSFA_slice_ptr = sfa_m_slice.iterator + local_sf_k
                        tAgSFA_slice = cute.make_tensor(
                            tAgSFA_slice_ptr, layout=cute.make_layout((4,))
                        )

                        # Smem: write to swizzled position
                        tAsSFA_slice_ptr = tAsSFA_ktile.iterator + 512 * SI
                        tAsSFA_slice = cute.make_tensor(
                            tAsSFA_slice_ptr, cute.make_layout((4,))
                        )

                        cute.copy_atom_call(
                            sfa_atom_copy,
                            tAgSFA_slice,
                            tAsSFA_slice,
                            pred=sfa_predicate_tensor,
                        )

                    # Increment in S→R→T→C order (S innermost, C outermost)
                    sfa_s_idx = sfa_s_idx + 1
                    if sfa_s_idx >= conv_S:
                        sfa_s_idx = 0
                        sfa_r_idx = sfa_r_idx + 1
                        if sfa_r_idx >= conv_R:
                            sfa_r_idx = 0
                            sfa_t_idx = sfa_t_idx + 1
                            if sfa_t_idx >= conv_T:
                                sfa_t_idx = 0
                                sfa_c_tile_idx = sfa_c_tile_idx + 1
                                if sfa_c_tile_idx >= c_tiles_per_fpos:
                                    sfa_c_tile_idx = 0

                    # Commit this iter's LDGSTS as one cp.async group and
                    # gate on ≤K inflight. After wait_group(K), the LDGSTS
                    # for the stage from K iters ago is guaranteed complete
                    # on every CTA in the cluster.
                    ab_sfa_pipeline.producer_cp_async_commit()
                    ab_sfa_pipeline.producer_cp_async_wait(sfa_shift_K)

                    # Shift-by-K: arrive on stage from K iters ago.
                    if k_tile >= sfa_shift_K:
                        ab_sfa_pipeline.producer_arrive_remote(pending_a)

                    # Advance ringbuffer (oldest <- 2nd oldest <- current).
                    pending_a = pending_b
                    pending_b = cur_stage

                    # Peek (try_wait) SFA buffer empty for next iteration
                    sfa_producer_state.advance()
                    peek_sfa_empty_status = cutlass.Boolean(1)
                    if sfa_producer_state.count < k_tile_cnt:
                        peek_sfa_empty_status = ab_sfa_pipeline.producer_try_acquire(
                            sfa_producer_state
                        )

                #
                # Shift-by-K tail: drain all inflight LDGSTS, then arrive on
                # the K stages whose arrives were deferred.
                #
                ab_sfa_pipeline.producer_cp_async_wait(0)
                if k_tile_cnt >= sfa_shift_K:
                    ab_sfa_pipeline.producer_arrive_remote(pending_a)
                if k_tile_cnt >= 1:
                    ab_sfa_pipeline.producer_arrive_remote(pending_b)

                #
                # Advance to next tile (CLC consumer)
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            #
            # Wait SFA buffer empty
            #
            ab_sfa_pipeline.producer_tail(sfa_producer_state)

        #
        # Specialized scheduler warp (drives CLC fetch, only first CTA in cluster)
        #
        if warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, self.num_clc_stage
            )

            tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                clc_pipeline.producer_acquire(clc_producer_state)
                mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
                tile_sched.advance_to_next_work(mbarrier_addr)
                clc_producer_state.advance()

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            clc_pipeline.producer_tail(clc_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # Make accumulator tmem tensor
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            #
            # Partition for S2T copy of SFA/SFB
            #
            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            sfa_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                # Overlapping-accum squeezes two acc buffers into 512-col TMEM by
                # aliasing the second buffer onto the SFA/SFB columns. The producer
                # writes the buffer the consumer is NOT currently draining, which is
                # acc_producer_state.phase ^ 1 (producer phase inits to 1, consumer to
                # 0, so tile 0 picks stage 0 to match the consumer). See TRTLLM
                # dense_blockscaled_gemm_persistent.py producer phase^1 indexing.
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # Peek (try_wait) shared AB/SFB + SFA buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                sfa_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                peek_sfa_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_sfa_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )
                # Merged pipeline: sfa shares ab mbarrier, no separate try_wait needed.

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acd_pipeline.producer_acquire(acc_producer_state)

                # Special-case for N=192/N=64: shift the SFB Tensor Memory pointer to match its packing
                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    # If this is an ODD tile, shift the Tensor Memory start address for cta_tile_shape_n=192 case by two words (ignores first 64 columns of SFB)
                    offset = (
                        cutlass.Int32(2)
                        if mma_tile_coord_mnl[1] % 2 == 1
                        else cutlass.Int32(0)
                    )
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                        + tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    # Move in increments of 64 columns of SFB
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                        + tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                    if is_leader_cta:
                        # Merged pipeline: single consumer_wait covers both TMA and cp.async producers.
                        ab_sfa_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        #  Copy SFA/SFB from smem to tmem
                        sfa_s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            sfa_consumer_state.index,
                        )
                        sfb_s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[
                            sfa_s2t_stage_coord
                        ]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[
                            sfb_s2t_stage_coord
                        ]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            # Set SFA/SFB tensor to tiled_mma
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB_mma[sf_kblock_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )

                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Merged pipeline: single consumer_release covers both TMA and cp.async producers.
                        ab_sfa_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) shared AB/SFB + SFA buffer full for k_tile+1
                    ab_consumer_state.advance()
                    sfa_consumer_state.advance()

                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_sfa_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )
                    peek_sfa_full_status = cutlass.Boolean(1)
                    # Merged pipeline: sfa shares ab mbarrier, try_wait above handles both.

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acd_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile (CLC consumer)
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            #
            # Wait for accumulator buffer empty
            #
            acd_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx <= self.epilogue_warp_id[-1]:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            #
            # Partition for epilogue
            #
            epi_tidx = tidx
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgD, epi_tile, use_2cta_instrs
            )

            tTR_rD = cute.make_rmem_tensor(tTR_rAcc.shape, self.d_dtype)
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rD, epi_tidx, sD
            )
            (
                tma_atom_d,
                bSG_sD,
                bSG_gD_partitioned,
            ) = self.epilog_gmem_copy_and_partition(
                epi_tidx, tma_atom_d, tCgD, epi_tile, sD
            )

            # SFD partition: same epi-tile/T2R structure as gD, but written
            # directly to GMEM via STG (no SMEM/TMA path).
            if cutlass.const_expr(self.gen_sfd):
                thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
                # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
                gSFD_epi = cute.flat_divide(
                    tCgSFD[((None, None), 0, 0, None, None, None)], epi_tile
                )
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
                tTR_gSFD_partitioned = thr_copy_t2r.partition_D(gSFD_epi)
                # The SFD store is a plain (unpredicated) STG. The cute
                # partition rounds M up to a tile multiple (RestM *
                # mma_tiler_m), which can exceed the real problem M
                # (conv_N*conv_Z*conv_P*conv_Q). On the last m-tile the
                # overhang rows have no backing storage, so an unguarded STG
                # would write past the SFD buffer into the physically-adjacent
                # allocation. Build an identity-coordinate mirror through the
                # D path (which has no stride-0 broadcast mode), partitioned
                # identically, so the store site can recover each thread's M
                # coordinate and predicate the STG against the real M extent.
                sfd_ceil_m, sfd_ceil_n, _ = cute.ceil_div(
                    mC_mnl.shape, (self.mma_tiler[0], self.mma_tiler[1], 1)
                )
                mcSFD = cute.make_identity_tensor(
                    (
                        cute.size(sfd_ceil_m) * self.mma_tiler[0],
                        cute.size(sfd_ceil_n) * self.mma_tiler[1],
                        1,
                    )
                )
                gcSFD = cute.local_tile(
                    mcSFD,
                    cute.slice_(self.mma_tiler, (None, None, 0)),
                    (None, None, None),
                )
                tCgcSFD = thr_mma.partition_C(gcSFD)
                gcSFD_epi = cute.flat_divide(
                    tCgcSFD[((None, None), 0, 0, None, None, None)], epi_tile
                )
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
                tTR_cSFD_partitioned = thr_copy_t2r.partition_D(gcSFD_epi)
            else:
                tTR_gSFD_partitioned = None
                tTR_cSFD_partitioned = None

            # Hoist 1-element FP32 device-tensor scalars OUTSIDE persistent
            # loop. Reading alpha[0] / norm_const_tensor[0] inside the
            # per-thread per-subtile loop returns garbage (JIT scalar tensor
            # read bug); reading them once here at the warp setup level works.
            alpha_scalar = cutlass.Float32(alpha[0])
            if cutlass.const_expr(self.gen_sfd):
                norm_const_scalar = cutlass.Float32(norm_const_tensor[0])
            else:
                norm_const_scalar = cutlass.Float32(1.0)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # Pipeline Init: Threads/warps participating in tma store pipeline
            d_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilogue_warp_id),
            )
            d_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_d_stage,
                producer_group=d_producer_group,
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                #
                # Slice to per mma tile index
                #
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gD = bSG_gD_partitioned[
                    (
                        None,
                        None,
                        None,
                        *mma_tile_coord_mnl,
                    )
                ]
                if cutlass.const_expr(self.gen_sfd):
                    # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
                    tTR_gSFD = tTR_gSFD_partitioned[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            *mma_tile_coord_mnl,
                        )
                    ]
                    # Identity-coordinate mirror, sliced identically, for the
                    # STG M-bound predicate at the SFD store site.
                    tTR_cSFD = tTR_cSFD_partitioned[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            *mma_tile_coord_mnl,
                        )
                    ]

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                # Overlapping-accum: the consumer reads the buffer indexed by its
                # phase (0/1). Stage-1 acc is strided into TMEM so that its low
                # subtiles alias the SFA/SFB columns; to drain those shared columns
                # first (before the next tile's MMA overwrites them) the epilogue
                # walks subtiles in REVERSE when phase==0. See TRTLLM
                # dense_blockscaled_gemm_persistent.py epilogue phase indexing.
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = acc_stage_index == 0
                else:
                    acc_stage_index = acc_consumer_state.index
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_stage_index)
                ]

                #
                # Wait for accumulator buffer full
                #
                acd_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gD = cute.group_modes(bSG_gD, 1, cute.rank(bSG_gD))
                if cutlass.const_expr(self.gen_sfd):
                    # ((T2R, T2R_M, T2R_N), SUBTILE_CNT)
                    tTR_gSFD = cute.group_modes(tTR_gSFD, 3, cute.rank(tTR_gSFD))
                    tTR_cSFD = cute.group_modes(tTR_cSFD, 3, cute.rank(tTR_cSFD))

                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in cutlass.range(subtile_cnt):
                    # Map the loop counter to the true output N-subtile. Under
                    # overlapping-accum with reverse_subtile we walk the output
                    # subtiles back-to-front so the SFA/SFB-aliased columns are
                    # drained (and released) before the next tile's MMA reuses
                    # them. real_subtile_idx addresses every actual output
                    # position (TMEM acc, gmem D, gmem SFD); the raw subtile_idx
                    # stays a sequential counter (SMEM ring, release).
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = (
                                self.cta_tile_shape_mnk[1] // self.epi_tile_n
                                - 1
                                - subtile_idx
                            )
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    #
                    # Async arrive accumulator buffer empty earlier when
                    # overlapping_accum is enabled. Trigger keyed on the raw loop
                    # counter so it fires exactly once after the shared columns
                    # (drained first under reverse) have been read out.
                    #
                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acd_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()

                    # Apply per-tensor alpha (hoisted scalar avoids JIT bug).
                    tTR_rAcc.store(tTR_rAcc.load() * alpha_scalar)

                    #
                    # SFD generation (NVFP4 output only): per sfd_vec_size
                    # abs-max -> pvscale_f32 -> cast to sf_dtype (E4M3) -> STG.
                    # Then rescale acc by norm_const * rcp(qpvscale_f32), with
                    # NaN/inf clamp via fmin. Matches TRT-LLM
                    # dense_blockscaled_gemm_swiglu_fusion encode pattern.
                    #
                    if cutlass.const_expr(self.gen_sfd):
                        # Slice gSFD for this subtile and collapse stride-0 broadcast.
                        gSFD_subtile = tTR_gSFD[(None, None, None, real_subtile_idx)]
                        t2r_gSFD = cute.filter_zeros(gSFD_subtile)
                        # M-bound predicate for the plain STG below. This
                        # thread's SFD fragment lies on a single M row (the
                        # fragment walks the N axis), so the first element's M
                        # coordinate gates the whole store. Rows in the last
                        # m-tile's overhang (m >= real M) have no backing
                        # storage and must be skipped to avoid an OOB STG.
                        # Predicate the plain STG on BOTH the M and N (channel)
                        # extents, using real_subtile_idx so the identity
                        # coordinate matches the actual gmem position written
                        # above (under overlapping-accum reverse_subtile the raw
                        # loop counter and the real output subtile are mirror
                        # images, so a raw-indexed predicate guards the wrong
                        # subtile). The cute partition rounds the cta tile up to a
                        # full mma_tiler multiple in both axes, so the last m-tile
                        # has overhang rows (m >= real M) and a cta_n that does not
                        # divide the real K-channel leaves an overhang N-subtile
                        # (n >= real K) on the partial tile. D goes through TMA, so
                        # its descriptor extent clamps both overhangs for free; the
                        # SFD store is a raw STG with no such clamp. An unguarded
                        # store of the N-overhang subtile writes quantized zeros
                        # (the overhang acc is out-of-bounds, TMA-clamped to 0) at a
                        # gmem offset that — because the mn-row stride equals the
                        # global sf_k count — folds onto the next row's lowest sf_k
                        # columns, zeroing valid scale factors there. The fragment
                        # lies on a single (m, n_base) coordinate, so the first
                        # element's coordinates gate the whole store.
                        cSFD_subtile = tTR_cSFD[(None, None, None, real_subtile_idx)]
                        sfd_m_in_bounds = cute.elem_less(
                            cSFD_subtile[0][0], mC_mnl.shape[0]
                        )
                        sfd_n_in_bounds = cute.elem_less(
                            cSFD_subtile[0][1], mC_mnl.shape[1]
                        )
                        sfd_in_bounds = sfd_m_in_bounds and sfd_n_in_bounds
                        # Partition tTR_rAcc into vec_size groups along the contig (K) mode.
                        sfgen_rAcc = cute.logical_divide(tTR_rAcc, self.sfd_vec_size)
                        n_sf = cute.size[1](sfgen_rAcc)
                        rSFD = cute.make_rmem_tensor((1, n_sf), dtype=self.sfd_dtype)
                        # Use hoisted scalar (avoids JIT scalar tensor read bug).
                        norm_const = norm_const_scalar
                        # 1/c_dtype_max for FP4 = 1/6.0.
                        rcp_max = cutlass.Float32(1.0 / self.M_D)
                        fp32_max = cutlass.Float32(3.40282346638528859812e38)
                        # Single-pass: compute amax, quantize SFD, and rescale acc
                        # in the same loop using a local f32 pvscale (no rmem
                        # tensor round-trip).
                        for i_sf in cutlass.range(n_sf, unroll_full=True):
                            sfgen_slice = sfgen_rAcc[(None, i_sf)]
                            red_ssa = sfgen_slice.load()
                            red_abs_ssa = cute.math.absf(red_ssa)
                            amax = cutlass.Float32(
                                red_abs_ssa.reduce(
                                    cute.ReductionOp.MAX,
                                    cutlass.Float32(0.0),
                                    0,
                                )
                            )
                            pvscale_f32 = amax * rcp_max * norm_const
                            rSFD[(0, i_sf)] = pvscale_f32.to(self.sfd_dtype)
                            acc_scale = norm_const * cute.arch.rcp_approx(pvscale_f32)
                            acc_scale = cutlass.Float32(
                                nvvm.fmin(
                                    acc_scale.ir_value(),
                                    fp32_max.ir_value(),
                                    nan=True,
                                )
                            )
                            sfgen_slice.store(sfgen_slice.load() * acc_scale)
                        # Store SFD to gmem (predicated on both the M and N
                        # bounds; the last m-tile and partial-N overhangs have no
                        # backing SFD storage).
                        if sfd_in_bounds:
                            if cutlass.const_expr(cute.size(rSFD) == 1):
                                t2r_gSFD[0] = rSFD[0]
                            else:
                                cute.autovec_copy(rSFD, t2r_gSFD)

                    #
                    # Convert to C type
                    #
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    acc_vec = epilogue_op(acc_vec.to(self.d_dtype))
                    tRS_rD.store(acc_vec)

                    #
                    # Store C to shared memory
                    #
                    d_buffer = (num_prev_subtiles + subtile_idx) % self.num_d_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD,
                        tRS_sD[(None, None, None, d_buffer)],
                    )
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    self.epilog_sync_barrier.arrive_and_wait()

                    #
                    # TMA store C to global memory
                    #
                    if warp_idx == self.epilogue_warp_id[0]:
                        cute.copy(
                            tma_atom_d,
                            bSG_sD[(None, d_buffer)],
                            bSG_gD[(None, real_subtile_idx)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        d_pipeline.producer_commit()
                        d_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty. Under overlapping-accum
                # the release already happened mid-loop (early release above), so
                # only the non-overlapping path releases here.
                #
                if cutlass.const_expr(not self.overlapping_accum):
                    with cute.arch.elect_one():
                        acd_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                #
                # Advance to next tile (CLC consumer)
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)
            #
            # Wait for C store complete
            #
            d_pipeline.producer_tail()

        # if bidx == 0 and bidy == 0 and bidz == 0:
        #     cute.printf(" tidx : {}, finished kernel", tidx)

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gD_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gD_mnl: The global tensor D
        :type gD_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.d_layout,
            self.d_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gD_mnl_epi = cute.flat_divide(
            gD_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gD = thr_copy_t2r.partition_D(gD_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gD[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rD: cute.Tensor,
        tidx: cutlass.Int32,
        sD: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rD: The partitioned accumulator tensor
        :type tTR_rD: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sD: The shared memory tensor to be copied and partitioned
        :type sD: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rD, tRS_sD) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rD: The partitioned tensor D (register source)
            - tRS_sD: The partitioned tensor D (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.d_layout, self.d_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD)
        # (R2S, R2S_M, R2S_N)
        tRS_rD = tiled_copy_r2s.retile(tTR_rD)
        return tiled_copy_r2s, tRS_rD, tRS_sD

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gD_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sD: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_d to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gD_mnl: The global tensor D
        :type gD_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sD: The shared memory tensor to be copied and partitioned
        :type sD: cute.Tensor

        :return: A tuple containing (tma_atom_d, bSG_sD, bSG_gD) where:
            - tma_atom_d: The TMA copy atom
            - bSG_sD: The partitioned shared memory tensor D
            - bSG_gD: The partitioned global tensor D
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gD_epi = cute.flat_divide(
            gD_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        tma_atom_d = atom
        sD_for_tma_partition = cute.group_modes(sD, 0, 2)
        gD_for_tma_partition = cute.group_modes(gD_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sD, bSG_gD = cpasync.tma_partition(
            tma_atom_d,
            0,
            cute.make_layout(1),
            sD_for_tma_partition,
            gD_for_tma_partition,
        )
        return tma_atom_d, bSG_sD, bSG_gD

    def _make_acc_fake_tensor(self, tiled_mma, mma_tiler):
        """Build the accumulator fake tensor used for TMEM column accounting.

        For overlapping-accum the second logical acc buffer is strided into the
        columns otherwise reserved for SFA/SFB (so the next tile's math can
        overlap the current tile's epilogue). The stride for the stage dim is
        (cta_tile_n - num_sf_tmem_cols) * stride[0][1], matching the TRTLLM
        NVFP4 GEMM trick. Otherwise fall back to a plain num_acc_stage fake.
        """
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        if cutlass.const_expr(self.overlapping_accum):
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 2))
            s = tCtAcc_fake.stride
            return cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        s[0],
                        s[1],
                        s[2],
                        (self.cta_tile_shape_mnk[1] - self.num_sf_tmem_cols) * s[0][1],
                    ),
                ),
            )
        return tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        d_dtype: Type[cutlass.Numeric],
        d_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param d_dtype: Data type of operand C (output).
        :type d_dtype: type[cutlass.Numeric]
        :param d_layout: Layout enum of operand C.
        :type d_layout: utils.LayoutEnum
        :param sf_dtype: Data type of Scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Scale factor vector size.
        :type sf_vec_size: int
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # ACC stages
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # Default C stages
        num_d_stage = 2

        # Calculate smem layout and size for one stage of A, B, SFA, SFB and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )
        # Flatten hierarchical K for blockscaled utils
        flat_k = (
            mma_tiler_mnk[2]
            if isinstance(mma_tiler_mnk[2], int)
            else mma_tiler_mnk[2][0]
        )
        flat_mma_tiler_mnk = (mma_tiler_mnk[0], mma_tiler_mnk[1], flat_k)
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            flat_mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            flat_mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        d_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            d_dtype,
            d_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024
        d_bytes_per_stage = cute.size_in_bytes(d_dtype, d_smem_layout_staged_one)
        d_bytes = d_bytes_per_stage * num_d_stage

        # Calculate A/B/SFA/SFB stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial D stages bytes
        # Divide remaining by bytes needed per A/B/SFA/SFB stage
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + d_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B/SFA/SFB stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_d_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + d_bytes)
        ) // (occupancy * d_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_d_stage

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        d_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes and sf_vec_size are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes and sf_vec_size are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # Check valid ab_dtype
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        # Check valid d_dtype
        if d_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
            cutlass.Float4E2M1FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        ab_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        d_major: str,
    ) -> bool:
        """
        Check if layouts and dtypes are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: str
        :param b_major: The major dimension of the B tensor
        :type b_major: str
        :param d_major: The major dimension of the D tensor
        :type d_major: str

        :return: True if the layouts are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        # TODO: Currently we don't support m major output for Float4E2M1FN
        if d_dtype is cutlass.Float4E2M1FN and d_major == "m":
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """
        Check if the mma tiler and cluster shape are valid

        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # Skip invalid mma tile shape
        if mma_tiler_mn[0] not in [128, 256]:
            is_valid = False
        if mma_tiler_mn[1] not in [64, 128, 192, 256]:
            is_valid = False
        # Skip illegal cluster shape
        if cluster_shape_mn[0] % (2 if mma_tiler_mn[0] == 256 else 1) != 0:
            is_valid = False
        # Skip invalid cluster shape
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            # Special cluster shape check for scale factor multicasts.
            # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        d_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the D tensor
        :type l: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param d_major: The major axis of the D tensor
        :type d_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(d_dtype, d_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid

    def can_implement(
        self,
        c: int,
        k: int,
        ab_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """Determine if the given tensor configuration can be implemented by this kernel.
        Version A (slice): supports C >= 64 (4 * sf_vec_size) for LDGSTS.32 alignment.
        """
        try:
            # FP4 dtype check (parent check_supported_dtypes doesn't include FP4)
            valid_ab = {cutlass.Float4E2M1FN, cutlass.Float8E5M2, cutlass.Float8E4M3FN}
            if ab_dtype not in valid_ab:
                raise testing.CantImplementError(f"Unsupported AB dtype: {ab_dtype}")
            self.check_mma_tiler_and_cluster_shape()
            _check_tensor_alignment(c, k, ab_dtype, d_dtype)
            # Version A constraint: C must be >= 4 * sf_vec_size (typically 64)
            # so that each filter position has at least 4 SF blocks for LDGSTS.32
            min_c = 4 * self.sf_vec_size
            if c < min_c:
                raise testing.CantImplementError(
                    f"Version A (slice) requires C >= {min_c}, but got C = {c}. "
                    f"Each filter position needs >= 4 SF blocks for LDGSTS.32 alignment."
                )
        except testing.CantImplementError as e:
            print(e)
            return False
        return True


@lru_cache(maxsize=1)
def compile_conv(
    ncdhw: Tuple[int, int, int, int, int],
    ktrs: Tuple[int, int, int, int],
    input: cute.Tensor,
    filter: cute.Tensor,
    output: cute.Tensor,
    acc_dtype: Type[cutlass.Numeric],
    sfa: cute.Tensor,
    sfb: cute.Tensor,
    sfd: Optional[cute.Tensor],
    alpha: cute.Tensor,
    norm_const_tensor: Optional[cute.Tensor],
    sf_vec_size: int,
    mma_tiler: Tuple[int, int] = (256, 256),
    preferred_cluster_shape_mn: Tuple[int, int] = (2, 1),
    fallback_cluster_shape_mn: Tuple[int, int] = (1, 1),
    swizzle_size: int = 1,
    raster_along: Literal["m", "n"] = "m",
    use_2cta_instrs: bool = True,
    upper_padding_dhw: Tuple[int, int, int] = (0, 0, 0),
    lower_padding_dhw: Tuple[int, int, int] = (0, 0, 0),
    stride_dhw: Tuple[int, int, int] = (1, 1, 1),
    dilation_dhw: Tuple[int, int, int] = (1, 1, 1),
    epilogue_op: cutlass.Constexpr = lambda x: x,
):
    """
    Compile a 3D convolution kernel.

    :param ncdhw: Problem shape (N, C, D, H, W)
    :param ktrs: Problem shape (K, T, R, S)
    :param input: Input tensor (N, D, H, W, C) with C contiguous
    :param filter: Filter tensor in KTRSC format (K, T, R, S, C) with C contiguous
    :param output: Output tensor (N, Z, P, Q, K) with K contiguous
    :param acc_dtype: Accumulator data type
    :param mma_tiler: MMA tile shape (M, N)
    :param preferred_cluster_shape_mn: Preferred cluster shape (M, N) for CLC dynamic scheduling
    :param fallback_cluster_shape_mn: Fallback cluster shape (M, N) for CLC dynamic scheduling
    :param swizzle_size: Swizzling size in the unit of cluster. 1 means no swizzle
    :param raster_along: Rasterization order of clusters. Only used when swizzle_size > 1
    :param use_2cta_instrs: Whether to use 2CTA instructions
    :param upper_padding_dhw: Upper padding (PadD, PadH, PadW)
    :param lower_padding_dhw: Lower padding (PadD, PadH, PadW)
    :param stride_dhw: Stride (Sd, Sh, Sw)
    :param dilation_dhw: Dilation (DilD, DilH, DilW)
    :param epilogue_op: Epilogue operation
    :return: Compiled kernel function
    """
    from cutlass.cute.runtime import make_fake_stream

    # Output spatial dims for the SFD global descriptor (host int, trace-const).
    zpq = compute_zpq(
        ncdhw[2:],
        ktrs[1:],
        stride_dhw,
        upper_padding_dhw,
        lower_padding_dhw,
        dilation_dhw,
    )
    output_nzpq = (ncdhw[0], zpq[0], zpq[1], zpq[2])

    # Create convolution kernel object. input_C, output_K, output_nzpq stay
    # host int (layout-shaping, trace-const). Filter T/R/S and pad/stride/dil
    # are NOT passed here: T/R/S come from the dynamic-layout filter tensor and
    # pad/stride/dil arrive as boxed runtime Int32 at the cute.compile entry.
    conv_op = Sm100BlockScaledPersistentDenseImplicitGemmKernel(
        acc_dtype,
        sf_vec_size,
        use_2cta_instrs,
        mma_tiler,
        preferred_cluster_shape_mn,
        fallback_cluster_shape_mn,
        ncdhw[1],
        ktrs[0],
        output_nzpq,
        swizzle_size,
        raster_along,
    )

    # Check if configuration can be implemented
    can_implement = conv_op.can_implement(
        ncdhw[1],
        ktrs[0],
        input.element_type,
        output.element_type,
    )
    if not can_implement:
        raise testing.CantImplementError("The current config is invalid/unsupported.")

    stream = make_fake_stream()
    # Box pad/stride/dilation as cutlass.Int32 so the cute.compile entry scalars
    # lower to runtime SSA AND keep their values out of the mangled function
    # name; that combination is what lets one cubin be reused across configs. A
    # raw Python int here would fold the value into the name and force
    # recompilation per config. The filter T/R/S are already runtime (taken from
    # the dynamic-layout filter tensor extents).
    return cute.compile(
        conv_op,
        input,
        filter,
        output,
        sfa,
        sfb,
        alpha,
        stream,
        epilogue_op,
        sfd,
        norm_const_tensor,
        cutlass.Int32(upper_padding_dhw[0]),
        cutlass.Int32(upper_padding_dhw[1]),
        cutlass.Int32(upper_padding_dhw[2]),
        cutlass.Int32(lower_padding_dhw[0]),
        cutlass.Int32(lower_padding_dhw[1]),
        cutlass.Int32(lower_padding_dhw[2]),
        cutlass.Int32(stride_dhw[0]),
        cutlass.Int32(stride_dhw[1]),
        cutlass.Int32(stride_dhw[2]),
        cutlass.Int32(dilation_dhw[0]),
        cutlass.Int32(dilation_dhw[1]),
        cutlass.Int32(dilation_dhw[2]),
    )


def compute_zpq(
    dhw: Tuple[int, int, int],
    trs: Tuple[int, int, int],
    stride_dhw: Tuple[int, int, int],
    upper_padding_dhw: Tuple[int, int, int],
    lower_padding_dhw: Tuple[int, int, int],
    dilation_dhw: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """Compute output spatial dimensions Z, P, and Q with asymmetric padding."""
    D, H, W = dhw
    T, R, S = trs
    Sd, Sh, Sw = stride_dhw
    UpperPadD, UpperPadH, UpperPadW = upper_padding_dhw
    LowerPadD, LowerPadH, LowerPadW = lower_padding_dhw
    DilD, DilH, DilW = dilation_dhw
    Z = ((D + UpperPadD + LowerPadD - DilD * (T - 1) - 1) // Sd) + 1
    P = ((H + UpperPadH + LowerPadH - DilH * (R - 1) - 1) // Sh) + 1
    Q = ((W + UpperPadW + LowerPadW - DilW * (S - 1) - 1) // Sw) + 1
    return Z, P, Q


def create_cute_tensor(
    source_f32_tensor: torch.Tensor,
    dtype: Type[cutlass.Numeric],
    leading_dim: int = None,
) -> Tuple[cute.Tensor, torch.Tensor]:
    """Create a dynamic-layout cute tensor from a source f32 tensor.

    The tensor is always marked dynamic-layout: its non-leading extents lower to
    runtime SSA so one compiled cubin serves any N/C/D/H/W/K/T/R/S config.

    :param source_f32_tensor: Source f32 tensor
    :type source_f32_tensor: torch.Tensor
    :param dtype: Data type
    :type dtype: Type[cutlass.Numeric]
    :param leading_dim: Leading dimension kept contiguous in the dynamic layout
    :type leading_dim: int
    :return: Tuple of cute tensor and storage tensor
    :rtype: Tuple[cute.Tensor, torch.Tensor]
    """

    # FP4 needs packed storage: 2 elements per byte. The shared cute_tensor_like
    # over-allocates 2x for FP4 (uint8 buffer with same shape as source);
    # quantization writes only first half, kernel reads back second-half garbage
    # via the FP4 layout (0.5 byte/elem stride).
    if dtype == cutlass.Float4E2M1FN:
        shape = tuple(source_f32_tensor.shape)
        assert shape[-1] % 2 == 0, (
            f"FP4 packed storage requires trailing dim even, got shape={shape}"
        )
        packed_shape = shape[:-1] + (shape[-1] // 2,)
        storage_int8 = torch.empty(packed_shape, dtype=torch.int8, device="cuda")
        storage_view = storage_int8.view(dtype=torch.float4_e2m1fn_x2)
        cute_tensor = from_dlpack(storage_view, assumed_align=16)
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)

        if source_f32_tensor.numel() > 0:
            f32_gpu = (
                source_f32_tensor.cuda()
                if source_f32_tensor.device.type == "cpu"
                else source_f32_tensor
            )
            f32_cute = from_dlpack(f32_gpu)
            f32_cute = f32_cute.mark_layout_dynamic(leading_dim=leading_dim)
            cute.testing.convert(f32_cute, cute_tensor)
        return cute_tensor, storage_view

    cute_tensor, storage_tensor = cutlass_torch.cute_tensor_like(
        source_f32_tensor, dtype, is_dynamic_layout=True, assumed_align=16
    )

    return cute_tensor, storage_tensor


def prepare_tensors(
    ncdhw: Tuple[int, int, int, int, int],
    ktrs: Tuple[int, int, int, int],
    zpq: Tuple[int, int, int],
    ab_dtype: Type[cutlass.Numeric],
):
    """Prepare f32 tensors for 3D convolution.

    :param ncdhw: Input tensor shape (N, C, D, H, W)
    :type ncdhw: Tuple[int, int, int, int, int]
    :param ktrs: Filter tensor shape components (K, T, R, S)
    :type ktrs: Tuple[int, int, int, int]
    :param zpq: Output spatial dimensions (Z, P, Q)
    :type zpq: Tuple[int, int, int]
    :param ab_dtype: Data type for A/B input tensors
    :type ab_dtype: Type[cutlass.Numeric]
    :return: Tuple of input, filter, and output tensors
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    N, C, D, H, W = ncdhw
    K, T, R, S = ktrs
    Z, P, Q = zpq

    # Initialize with small random values for numerical stability
    if ab_dtype == cutlass.Uint8:
        input_range = (0, 2)
    else:
        input_range = (-1, 2)
    input_tensor = torch.randint(
        input_range[0],
        input_range[1],
        (N, D, H, W, C),
        dtype=torch.float32,
        device="cuda",
    )
    filter_tensor = torch.randint(
        input_range[0],
        input_range[1],
        (K, T, R, S, C),
        dtype=torch.float32,
        device="cuda",
    )
    output_tensor = torch.empty((N, Z, P, Q, K), dtype=torch.float32, device="cuda")

    return input_tensor, filter_tensor, output_tensor


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


def run(
    ncdhw: Tuple[int, int, int, int, int],
    ktrs: Tuple[int, int, int, int],
    stride_dhw: Tuple[int, int, int] = (1, 1, 1),
    upper_pad_dhw: Tuple[int, int, int] = (0, 0, 0),
    lower_pad_dhw: Tuple[int, int, int] = (0, 0, 0),
    dil_dhw: Tuple[int, int, int] = (1, 1, 1),
    ab_dtype: Type[cutlass.Numeric] = cutlass.Float4E2M1FN,
    d_dtype: Type[cutlass.Numeric] = cutlass.Float4E2M1FN,
    acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
    sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
    sf_vec_size: int = 16,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    preferred_cluster_shape_mn: Tuple[int, int] = (2, 1),
    fallback_cluster_shape_mn: Tuple[int, int] = (1, 1),
    swizzle_size: int = 1,
    raster_along: Literal["m", "n"] = "m",
    use_2cta_instrs: bool = False,
    tolerance: float = 1e-02,
    warmup_iterations: int = 0,
    iterations: int = 1,
    use_cold_l2: bool = False,
    skip_ref_check: bool = False,
    **kwargs,
):
    """Run 3D convolution and compare against PyTorch reference.

    The filter tensor uses native KTRSC layout (K, T, R, S, C) with C contiguous.

    :param ncdhw: Input tensor shape (N, C, D, H, W)
    :param ktrs: Filter tensor shape components (K, T, R, S)
    :param stride_dhw: Stride (Sd, Sh, Sw)
    :param upper_pad_dhw: Upper padding (PadD, PadH, PadW)
    :param lower_pad_dhw: Lower padding (PadD, PadH, PadW)
    :param dil_dhw: Dilation (DilD, DilH, DilW)
    :param ab_dtype: Data type for A/B input tensors
    :param d_dtype: Data type for output tensor D
    :param acc_dtype: Accumulator data type
    :param mma_tiler_mn: MMA tiler shape
    :param preferred_cluster_shape_mn: Preferred cluster shape (M, N) for CLC dynamic scheduling
    :param fallback_cluster_shape_mn: Fallback cluster shape (M, N) for CLC dynamic scheduling
    :param swizzle_size: Swizzling size in the unit of cluster. 1 means no swizzle
    :param raster_along: Rasterization order of clusters
    :param use_2cta_instrs: Whether to use 2CTA instructions
    :param tolerance: Tolerance for result comparison
    :param warmup_iterations: Number of warmup iterations
    :param iterations: Number of benchmark iterations
    :param use_cold_l2: Whether to flush L2 cache between iterations
    :param skip_ref_check: Whether to skip reference checking
    """
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    N, C, D, H, W = ncdhw
    K, T, R, S = ktrs

    Z, P, Q = compute_zpq(
        (D, H, W),
        (T, R, S),
        stride_dhw,
        upper_pad_dhw,
        lower_pad_dhw,
        dil_dhw,
    )

    print("Running Blackwell 3D Convolution test with:")
    print(f"  Input shape (N, C, D, H, W): {ncdhw}")
    print(f"  Filter shape (K, C, T, R, S): ({K}, {C}, {T}, {R}, {S})")
    print(f"  Output shape (N, K, Z, P, Q): ({N}, {K}, {Z}, {P}, {Q})")
    print(f"  Stride (Sd, Sh, Sw): {stride_dhw}")
    print(f"  Upper padding (PadD, PadH, PadW): {upper_pad_dhw}")
    print(f"  Lower padding (PadD, PadH, PadW): {lower_pad_dhw}")
    print(f"  Dilation (DilD, DilH, DilW): {dil_dhw}")
    print(f"  A/B data type: {ab_dtype}")
    print(f"  D data type: {d_dtype}")
    print(f"  Accumulator type: {acc_dtype}")
    print(f"  sf data type: {sf_dtype}")
    print(f"  sf vec size: {sf_vec_size}")
    print(f"  MMA tiler (M, N): {mma_tiler_mn}")
    print(f"  Preferred cluster shape (M, N): {preferred_cluster_shape_mn}")
    print(f"  Fallback cluster shape (M, N): {fallback_cluster_shape_mn}")
    print(f"  Swizzle size: {swizzle_size}")
    print(f"  Raster along: {raster_along}")
    print(f"  Use 2CTA instructions: {use_2cta_instrs}")
    print()

    # Create input and filter tensors
    input_tensor, filter_tensor, output_tensor = prepare_tensors(
        ncdhw, ktrs, (Z, P, Q), ab_dtype
    )

    # Prepare cute tensors
    input_, input_storage = create_cute_tensor(input_tensor, ab_dtype, leading_dim=4)
    filter_, filter_storage = create_cute_tensor(filter_tensor, ab_dtype, leading_dim=4)
    output_, output_storage = create_cute_tensor(output_tensor, d_dtype, leading_dim=4)

    # Create scale factor tensor SFA/SFB
    def create_scale_factor_tensor_swizzled(l, mn, k, sf_vec_size, dtype):
        def ceil_div(a, b):
            return (a + b - 1) // b

        sf_k = ceil_div(k, sf_vec_size)
        ref_shape = (l, mn, sf_k)

        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            l,
            ceil_div(mn, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )

        ref_permute_order = (1, 2, 0)
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        # Create f32 ref torch tensor (cpu)
        ref_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            ref_shape,
            torch.float32,
            permute_order=ref_permute_order,
            init_type=cutlass_torch.TensorInitType.RANDOM,
            init_config=cutlass_torch.RandomInitConfig(
                min_val=1,
                max_val=3,
            ),
        )
        # Create f32 cute torch tensor (cpu)
        cute_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            mma_shape,
            torch.float32,
            permute_order=mma_permute_order,
            init_type=cutlass_torch.TensorInitType.RANDOM,
            init_config=cutlass_torch.RandomInitConfig(
                min_val=0,
                max_val=1,
            ),
        )

        # convert ref f32 tensor to cute f32 tensor
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
            from_dlpack(ref_f32_torch_tensor_cpu),
            from_dlpack(cute_f32_torch_tensor_cpu),
        )
        cute_f32_torch_tensor = cute_f32_torch_tensor_cpu.cuda()

        # reshape makes memory contiguous
        ref_f32_torch_tensor_cpu = (
            ref_f32_torch_tensor_cpu.permute(2, 0, 1)
            .unsqueeze(-1)
            .expand(l, mn, sf_k, sf_vec_size)
            .reshape(l, mn, sf_k * sf_vec_size)
            .permute(*ref_permute_order)
        )
        # prune to mkl for reference check.
        ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]

        # Create dtype cute torch tensor (cpu)
        cute_tensor, cute_torch_tensor = cutlass_torch.cute_tensor_like(
            cute_f32_torch_tensor_cpu,
            dtype,
            is_dynamic_layout=True,
            assumed_align=16,
        )

        # Convert f32 cute tensor to dtype cute tensor
        cute_tensor = cutlass_torch.convert_cute_tensor(
            cute_f32_torch_tensor,
            cute_tensor,
            dtype,
            is_dynamic_layout=True,
        )
        return ref_f32_torch_tensor_cpu, cute_tensor, cute_torch_tensor

    def create_scale_factor_tensor_unswizzled(l, mn, k, sf_vec_size, dtype):
        def ceil_div(a, b):
            return (a + b - 1) // b

        sf_k = ceil_div(k, sf_vec_size)
        # Pad sf_k to multiple of 4 so that the M-direction stride (= sf_k_padded)
        # satisfies LDGSTS.32's 4-byte alignment requirement.
        atom_k = 4
        sf_k_padded = ceil_div(sf_k, atom_k) * atom_k

        # Use pure PyTorch to create the tensor, bypassing cute_tensor_like which
        # requires the leading mode to be divisible by 4 for 8-bit types.
        sf_raw = torch.randint(0, 3, (l, mn, sf_k_padded), dtype=torch.uint8).permute(
            1, 2, 0
        )
        if sf_k_padded > sf_k:
            sf_raw[:, sf_k:, :] = 0

        sf_torch = sf_raw.to(dtype=cutlass_torch.dtype(dtype)).cuda()
        sf_tensor = from_dlpack(sf_torch, assumed_align=16)

        # Build f32 reference for verification (only the original sf_k, not padded)
        sf_ref = (
            sf_raw[:, :sf_k, :]
            .float()
            .permute(2, 0, 1)
            .unsqueeze(-1)
            .expand(l, mn, sf_k, sf_vec_size)
            .reshape(l, mn, sf_k * sf_vec_size)
            .permute(1, 2, 0)
        )
        sf_ref = sf_ref[:, :k, :]

        return sf_ref, sf_tensor, sf_torch

    sfa_ref, sfa_, sfa_storage = create_scale_factor_tensor_unswizzled(
        1,
        N * D * H * W,
        C,
        sf_vec_size,
        sf_dtype,
    )
    sfb_ref, sfb_, sfb_storage = create_scale_factor_tensor_swizzled(
        1,
        K,
        C * T * R * S,
        sf_vec_size,
        sf_dtype,
    )
    # SFD: only emitted for NVFP4 (Float4E2M1FN) output, matching TRT-LLM
    # dense_blockscaled_gemm convention. FP8/FP16/BF16/FP32 outputs do not
    # carry SFD.
    gen_sfd = d_dtype is cutlass.Float4E2M1FN
    sfd_vec_size = 16
    # SFD shares the input sf_dtype (NVFP4 standard: Float8E4M3FN).
    sfd_dtype = sf_dtype
    if gen_sfd:
        _, sfd_, sfd_storage = create_scale_factor_tensor_unswizzled(
            1,
            N * Z * P * Q,
            K,
            sfd_vec_size,
            sfd_dtype,
        )
        # Helper randomizes by default — we want to read what the kernel writes.
        sfd_storage.zero_()
    else:
        sfd_, sfd_storage = None, None

    # Per-tensor FP32 scaling factors (TRT-LLM convention):
    #   alpha:            applied in accumulator precision before quantize
    #   norm_const_tensor: folded into SFD scale-up (only when gen_sfd)
    alpha_storage = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    alpha_ = from_dlpack(alpha_storage, assumed_align=4)
    if gen_sfd:
        norm_const_storage = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        norm_const_ = from_dlpack(norm_const_storage, assumed_align=4)
    else:
        norm_const_storage, norm_const_ = None, None

    # Compile convolution kernel
    print("Compiling kernel with cute.compile ...")
    compiled_fn = compile_conv(
        ncdhw,
        ktrs,
        input_,
        filter_,
        output_,
        acc_dtype,
        sfa_,
        sfb_,
        sfd_,
        alpha_,
        norm_const_,
        sf_vec_size,
        mma_tiler=mma_tiler_mn,
        preferred_cluster_shape_mn=preferred_cluster_shape_mn,
        fallback_cluster_shape_mn=fallback_cluster_shape_mn,
        swizzle_size=swizzle_size,
        raster_along=raster_along,
        use_2cta_instrs=use_2cta_instrs,
        upper_padding_dhw=upper_pad_dhw,
        lower_padding_dhw=lower_pad_dhw,
        stride_dhw=stride_dhw,
        dilation_dhw=dil_dhw,
    )

    # Get current CUDA stream
    torch_stream = torch.cuda.Stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    # The compiled entry expects the 12 pad/stride/dilation runtime scalars
    # (upper d/h/w, lower d/h/w, stride d/h/w, dil d/h/w) matching the boxed
    # cutlass.Int32 params baked into cute.compile; pass them as the same host
    # config values so one cubin serves any config.
    print("Running Blackwell 3D convolution...")
    compiled_fn(
        input_,
        filter_,
        output_,
        sfa_,
        sfb_,
        alpha_,
        current_stream,
        sfd_,
        norm_const_,
        cutlass.Int32(upper_pad_dhw[0]),
        cutlass.Int32(upper_pad_dhw[1]),
        cutlass.Int32(upper_pad_dhw[2]),
        cutlass.Int32(lower_pad_dhw[0]),
        cutlass.Int32(lower_pad_dhw[1]),
        cutlass.Int32(lower_pad_dhw[2]),
        cutlass.Int32(stride_dhw[0]),
        cutlass.Int32(stride_dhw[1]),
        cutlass.Int32(stride_dhw[2]),
        cutlass.Int32(dil_dhw[0]),
        cutlass.Int32(dil_dhw[1]),
        cutlass.Int32(dil_dhw[2]),
    )

    if not skip_ref_check:
        print("Verifying results with block-scaled reference...")
        # Block-scaled reference using F.conv3d with pre-scaled inputs
        # This handles padding/stride/dilation correctly

        # Pre-scale input by SFA: sfa_ref shape (N*D*H*W, C, 1)
        M_input = N * D * H * W
        sfa_expanded = sfa_ref.squeeze(-1).reshape(N, D, H, W, C).cuda()
        scaled_input = input_tensor.cuda().float() * sfa_expanded

        # Pre-scale filter by SFB: sfb_ref shape (K, C*T*R*S, 1)
        sfb_expanded = sfb_ref.squeeze(-1).reshape(K, T, R, S, C).cuda()
        scaled_filter = filter_tensor.cuda().float() * sfb_expanded

        # F.conv3d expects (N, C, D, H, W) and (K, C, T, R, S)
        scaled_input_ncdhw = scaled_input.permute(0, 4, 1, 2, 3).contiguous()
        scaled_filter_kctrs = scaled_filter.permute(0, 4, 1, 2, 3).contiguous()

        if upper_pad_dhw == lower_pad_dhw:
            ref_nkzpq = F.conv3d(
                scaled_input_ncdhw,
                scaled_filter_kctrs,
                padding=upper_pad_dhw,
                stride=stride_dhw,
                dilation=dil_dhw,
            )
        else:
            # Asymmetric padding: F.pad then conv3d without padding.
            # Convention: lower_pad = leading (left), upper_pad = trailing (right).
            # This must match the im2col TMA descriptor which subtracts lower_pad
            # when mapping output coords to input coords (d_in = z*stride - lower_pad + t*dil).
            padded = F.pad(
                scaled_input_ncdhw,
                (
                    lower_pad_dhw[2],
                    upper_pad_dhw[2],
                    lower_pad_dhw[1],
                    upper_pad_dhw[1],
                    lower_pad_dhw[0],
                    upper_pad_dhw[0],
                ),
            )
            ref_nkzpq = F.conv3d(
                padded,
                scaled_filter_kctrs,
                stride=stride_dhw,
                dilation=dil_dhw,
            )

        # Convert to NZPQK layout (matching kernel output)
        ref = ref_nkzpq.permute(0, 2, 3, 4, 1).contiguous()  # (N, Z, P, Q, K)
        # Snapshot the un-quantized FP32 ref BEFORE the in-place quantize round-trip
        # below mutates `ref` (shares GPU storage with ref_device).
        ref_unquant_cpu = ref.detach().cpu().clone()

        # Convert kernel FP4 output to f32 for comparison
        d_ref_device = torch.empty((N, Z, P, Q, K), dtype=torch.float32, device="cuda")
        cute.testing.convert(
            output_,
            from_dlpack(d_ref_device, assumed_align=16).mark_layout_dynamic(
                leading_dim=4
            ),
        )
        d_ref_result = d_ref_device.cpu()

        # Quantize reference: f32 -> d_dtype -> f32 (mutates ref in-place via shared
        # storage). Scratch storage dtype follows the kernel-convert rule: sub-byte
        # FP4 and <=8-bit floats pack into a uint8 byte buffer, while wider types
        # (f16/bf16/f32) use their native torch dtype so the buffer is not
        # under-allocated (f16 needs 2 bytes/elem, not 1).
        ref_quant_byte_storage = (d_dtype.is_float and d_dtype.width <= 8) or (
            d_dtype.is_integer and d_dtype.width == 4
        )
        ref_quant_storage_dtype = (
            torch.uint8 if ref_quant_byte_storage else cutlass_torch.dtype(d_dtype)
        )
        ref_f4_ = torch.empty(
            (N, Z, P, Q, K), dtype=ref_quant_storage_dtype, device="cuda"
        )
        ref_f4 = from_dlpack(ref_f4_, assumed_align=16).mark_layout_dynamic(
            leading_dim=4
        )
        ref_f4.element_type = d_dtype
        ref_device = ref.contiguous().cuda()
        ref_tensor = from_dlpack(ref_device, assumed_align=16).mark_layout_dynamic(
            leading_dim=4
        )
        cute.testing.convert(ref_tensor, ref_f4)
        cute.testing.convert(ref_f4, ref_tensor)
        ref_quantized = ref_device.cpu()

        if gen_sfd:
            # SFD path: kernel rescales acc by acc_scale before FP4 cast, so
            # raw kernel D is no longer comparable to a directly-quantized ref.
            # Do a dequant-equation refcheck: D_kernel * SFD ≈ ref (un-quantized).
            sfd_cpu = sfd_storage.cpu()
            sf_k = (K + sfd_vec_size - 1) // sfd_vec_size
            print(f"sfd_storage dtype = {sfd_cpu.dtype} (kernel sfd_dtype={sfd_dtype})")
            # Bit-reinterpret storage as raw uint8 bytes (for byte stats only).
            sfd_bytes_view = sfd_cpu.view(torch.uint8)
            # Reinterpret as the kernel sfd_dtype, then cast to float for dequant.
            if sfd_dtype is cutlass.Float8E4M3FN:
                sfd_typed_view = sfd_bytes_view.view(torch.float8_e4m3fn)
            else:
                sfd_typed_view = sfd_bytes_view.view(torch.float8_e8m0fnu)
            # Layout (mn=NZPQ, sf_k_padded, l=1) — keep only the live K range.
            sfd_dequant = (
                sfd_typed_view[:, :sf_k, :].float().squeeze(-1)
            )  # (NZPQ, sf_k)
            sfd_expanded = (
                sfd_dequant.unsqueeze(-1)
                .expand(-1, -1, sfd_vec_size)
                .reshape(-1, sf_k * sfd_vec_size)[:, :K]
                .reshape(N, Z, P, Q, K)
            )
            d_dequant = d_ref_result * sfd_expanded
            ref_cpu = ref_unquant_cpu
            n_nonzero = (sfd_bytes_view[:, :sf_k, :] != 0).sum().item()
            n_total = sfd_bytes_view[:, :sf_k, :].numel()
            print(
                f"SFD: {n_nonzero}/{n_total} entries non-zero; "
                f"D_dequant max={d_dequant.abs().max().item():.4g}, "
                f"ref max={ref_cpu.abs().max().item():.4g}"
            )
            assert n_nonzero > 0, "SFD is all zero — kernel did not write SFD"
            sfd_bytes = sfd_bytes_view[:, :sf_k, :].flatten()
            print(
                f"SFD bytes: min={sfd_bytes.min().item()} max={sfd_bytes.max().item()} "
                f"mean={sfd_bytes.float().mean().item():.2f}"
            )
            print(
                f"D_kernel:  min={d_ref_result.min().item():.4g} max={d_ref_result.max().item():.4g}"
            )
            print(
                f"ref:       min={ref_cpu.min().item():.4g} max={ref_cpu.max().item():.4g}"
            )
            # Strict dequant refcheck: D_kernel * SFD ≈ ref within FP4 round-off.
            # FP4 (Float4E2M1FN) values are {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} with
            # max spacing = 2 (between 4 and 6), so max round-to-nearest error per
            # element after dequant is `1.0 * sfd_scale`. Allow a small slack (1.5x)
            # for the e8m0 round-up bias.
            diff = (d_dequant - ref_cpu).abs()
            err_in_steps = diff / sfd_expanded.clamp(min=1e-6)
            print(
                f"\nD_dequant vs ref: max abs diff={diff.max().item():.4g}, "
                f"max err_in_steps={err_in_steps.max().item():.4g}, "
                f"mean abs diff={diff.mean().item():.4g}, "
                f"frac err_in_steps>1.5={(err_in_steps > 1.5).float().mean().item():.4f}"
            )
            assert err_in_steps.max().item() < 1.5, (
                f"SFD dequant exceeds FP4 round-off: max err_in_steps="
                f"{err_in_steps.max().item():.4g} > 1.5"
            )
            print("SFD strict dequant refcheck passed.")
        else:
            torch.testing.assert_close(
                d_ref_result,
                ref_quantized,
                atol=tolerance,
                rtol=1e-02,
            )
            print("Results match within tolerance!")

    # Benchmark if requested
    if iterations > 0:
        print(
            f"\nBenchmarking with {warmup_iterations} warmup and {iterations} iterations..."
        )

        def generate_tensors():
            input_tensor, filter_tensor, output_tensor = prepare_tensors(
                ncdhw, ktrs, (Z, P, Q), ab_dtype
            )
            input_, input_storage = create_cute_tensor(
                input_tensor, ab_dtype, leading_dim=4
            )
            filter_, filter_storage = create_cute_tensor(
                filter_tensor, ab_dtype, leading_dim=4
            )
            output_, output_storage = create_cute_tensor(
                output_tensor, d_dtype, leading_dim=4
            )
            # Arg order must match the compiled entry exactly (epilogue_op is a
            # Constexpr, not a runtime arg): a, b, d, sfa, sfb, alpha, stream,
            # sfd, norm_const, then the 12 pad/stride/dil runtime Int32. Only
            # the A/B/D tensors rotate per cold-L2 workspace; the SF tensors,
            # alpha/norm_const, and the geometry scalars are reused.
            return testing.JitArguments(
                input_,
                filter_,
                output_,
                sfa_,
                sfb_,
                alpha_,
                current_stream,
                sfd_,
                norm_const_,
                cutlass.Int32(upper_pad_dhw[0]),
                cutlass.Int32(upper_pad_dhw[1]),
                cutlass.Int32(upper_pad_dhw[2]),
                cutlass.Int32(lower_pad_dhw[0]),
                cutlass.Int32(lower_pad_dhw[1]),
                cutlass.Int32(lower_pad_dhw[2]),
                cutlass.Int32(stride_dhw[0]),
                cutlass.Int32(stride_dhw[1]),
                cutlass.Int32(stride_dhw[2]),
                cutlass.Int32(dil_dhw[0]),
                cutlass.Int32(dil_dhw[1]),
                cutlass.Int32(dil_dhw[2]),
            )

        workspace_count = 1
        if use_cold_l2:
            one_workspace_bytes = (
                input_storage.numel() * input_storage.element_size()
                + filter_storage.numel() * filter_storage.element_size()
                + output_storage.numel() * output_storage.element_size()
                + sfa_storage.numel() * sfa_storage.element_size()
                + sfb_storage.numel() * sfb_storage.element_size()
            )
            workspace_count = testing.get_workspace_count(
                one_workspace_bytes, warmup_iterations, iterations
            )

        # exec_time is in microseconds
        exec_time = testing.benchmark(
            compiled_fn,
            workspace_generator=generate_tensors,
            workspace_count=workspace_count,
            stream=current_stream,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
            use_cuda_graphs=True,
        )
        runtime_s = exec_time / 1.0e6
        fmas = (N * Z * P * Q) * K * (C * T * R * S)
        flop = 2 * fmas
        gflop = flop / 1.0e9
        gflops = gflop / runtime_s

        print("Average Runtime : ", exec_time / 1000, "ms")
        print("GFLOPS          : ", gflops)

        return exec_time


def _parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blackwell 3D convolution")

    # Convolution parameters
    parser.add_argument(
        "--ncdhw",
        type=_parse_comma_separated_ints,
        default=(1, 64, 3, 3, 3),
        help="Input tensor shape (N,C,D,H,W)",
    )
    parser.add_argument(
        "--ktrs",
        type=_parse_comma_separated_ints,
        default=(128, 3, 3, 3),
        help="Filter tensor shape components (K,T,R,S)",
    )
    parser.add_argument(
        "--stride_dhw",
        type=_parse_comma_separated_ints,
        default=(1, 1, 1),
        help="Stride (Sd,Sh,Sw)",
    )
    parser.add_argument(
        "--upper_pad_dhw",
        type=_parse_comma_separated_ints,
        default=(1, 1, 1),
        help="Upper padding (PadD,PadH,PadW)",
    )
    parser.add_argument(
        "--lower_pad_dhw",
        type=_parse_comma_separated_ints,
        default=(1, 1, 1),
        help="Lower padding (PadD,PadH,PadW)",
    )
    parser.add_argument(
        "--dil_dhw",
        type=_parse_comma_separated_ints,
        default=(1, 1, 1),
        help="Dilation (DilD,DilH,DilW)",
    )

    # Data type parameters
    parser.add_argument(
        "--ab_dtype",
        type=cutlass.dtype,
        choices=[
            cutlass.Float4E2M1FN,
        ],
        default=cutlass.Float4E2M1FN,
        help="Data type for A/B input tensors",
    )
    parser.add_argument(
        "--d_dtype",
        type=cutlass.dtype,
        choices=[
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float32,
        ],
        default=cutlass.Float4E2M1FN,
        help="Output D data type. SFD is generated only when width <= 8.",
    )
    parser.add_argument(
        "--c_dtype",
        type=cutlass.dtype,
        default=None,
        help="DEPRECATED alias for --d_dtype.",
    )
    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        choices=[cutlass.Float32, cutlass.Float16, cutlass.Int32],
        default=cutlass.Float32,
        help="Accumulator data type",
    )
    parser.add_argument(
        "--sf_dtype",
        type=cutlass.dtype,
        choices=[
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
        ],
        default=cutlass.Float8E4M3FN,
        help="Data type for A/B/D scaling factor tensors (NVFP4 default: E4M3)",
    )
    parser.add_argument("--sf_vec_size", type=int, default=16)

    # Kernel parameters
    parser.add_argument(
        "--mma_tiler_mn",
        type=_parse_comma_separated_ints,
        default=(128, 128),
        help="MMA tiler shape (M,N)",
    )
    parser.add_argument(
        "--preferred_cluster_shape_mn",
        type=_parse_comma_separated_ints,
        default=(2, 1),
        help="Preferred cluster shape (M,N) for CLC dynamic scheduling",
    )
    parser.add_argument(
        "--fallback_cluster_shape_mn",
        type=_parse_comma_separated_ints,
        default=(1, 1),
        help="Fallback cluster shape (M,N) for CLC dynamic scheduling",
    )
    parser.add_argument(
        "--use_2cta_instrs",
        action="store_true",
        help="Enable 2CTA MMA instructions",
    )
    parser.add_argument(
        "--swizzle_size",
        type=int,
        default=1,
        help="Swizzling size in the unit of cluster for improving L2 cache hit rate",
    )
    parser.add_argument(
        "--raster_order",
        type=str,
        choices=["m", "n"],
        default="m",
        help="Rasterization order of clusters",
    )
    # Testing parameters
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-02,
        help="Tolerance for result comparison",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=0,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--skip_ref_check",
        action="store_true",
        help="Skip reference checking",
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )
    args = parser.parse_args()
    if args.c_dtype is not None:
        print("WARNING: --c_dtype is deprecated; use --d_dtype.")
        args.d_dtype = args.c_dtype

    run(
        args.ncdhw,
        args.ktrs,
        args.stride_dhw,
        args.upper_pad_dhw,
        args.lower_pad_dhw,
        args.dil_dhw,
        args.ab_dtype,
        args.d_dtype,
        args.acc_dtype,
        args.sf_dtype,
        args.sf_vec_size,
        args.mma_tiler_mn,
        args.preferred_cluster_shape_mn,
        args.fallback_cluster_shape_mn,
        args.swizzle_size,
        args.raster_order,
        args.use_2cta_instrs,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.use_cold_l2,
        args.skip_ref_check,
    )
    print("PASS")
