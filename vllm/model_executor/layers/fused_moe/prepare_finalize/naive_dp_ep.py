# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Generator
from contextlib import contextmanager

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input,
    restore_dispatched_scale_layout,
)
from vllm.platforms import current_platform
from vllm.v1.worker.ubatching import (
    dbo_enabled,
    dbo_switch_to_compute_sync,
    dbo_yield_and_switch_from_compute_to_comm,
)


def _quantize_and_setup_dispatch(
    a1: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    defer_input_quant: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor | None]:
    # Defer input quantization to the MoE kernel.
    if defer_input_quant:
        a1q = a1
        a1q_scale = None
    else:
        input_sf = (
            quant_config.a1_gscale
            if quant_config.use_nvfp4_w4a4
            else quant_config.a1_scale
        )

        # NOTE: swizzling pads the scales to multiple of 128
        # which makes the scales tensor different shape than
        # the hidden states, breaking the A2A kernel. So, we
        # delay the swizzling until after the A2A.
        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            input_sf,
            quant_dtype=quant_config.quant_dtype,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=quant_config.block_shape,
            is_scale_swizzled=False,
            mx_alignment=quant_config.mx_alignment,
        )

    # Skip gathering scales if we have static quantization
    # (the scale is a scalar, replicated on all ranks) or
    # if quantization is deferred.
    skip_gather_scales = a1q_scale is None or a1q_scale.ndim == 0
    scales = None if skip_gather_scales else [a1q_scale]

    return a1q, scales, a1q_scale


def _localize_shared_expert_ids(topk_ids: torch.Tensor) -> torch.Tensor:
    if not current_platform.is_rocm():
        return topk_ids
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        localize_fused_shared_expert_ids,
    )

    return localize_fused_shared_expert_ids(topk_ids)


def _num_dispatched_tokens() -> int | None:
    """Rows the dispatch all-gathers, summed over the ranks; None when the DP
    chunk sizes are not known."""
    if not is_forward_context_available():
        return None
    dp_metadata = get_forward_context().dp_metadata
    if dp_metadata is None or dp_metadata.local_sizes is None:
        return None
    return sum(dp_metadata.local_sizes)


def _use_mxfp4_dispatch(
    a1: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    defer_input_quant: bool,
) -> bool:
    """Whether to quantize to MXFP4 before the all-gather for experts that
    would quantize the gathered rows themselves (ROCm AITER, see
    `use_mxfp4_dispatch`)."""
    if not (
        current_platform.is_rocm()
        and defer_input_quant
        and quant_config.use_mxfp4_w4a4
        and a1.dtype == torch.bfloat16
        and a1.dim() == 2
    ):
        return False
    num_tokens = _num_dispatched_tokens()
    if num_tokens is None:
        return False
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        use_mxfp4_dispatch,
    )

    return use_mxfp4_dispatch(num_tokens, a1.shape[1])


def _packed_dispatch_manager():
    """The EP group's all-gather/reduce-scatter all2all manager, which can
    gather a packed dispatch buffer; None for any other manager."""
    from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager

    manager = getattr(get_ep_group().device_communicator, "all2all_manager", None)
    return manager if isinstance(manager, AgRsAll2AllManager) else None


def _unwrap_scale_and_prepare_for_moe(
    scales: list[torch.Tensor] | None,
    quant_config: FusedMoEQuantConfig,
) -> torch.Tensor:
    assert scales is not None and len(scales) == 1
    # Apply swizzling after a2a if the MoE kernel needs it.
    a1q_scale = restore_dispatched_scale_layout(
        scales[0], quant_config.quant_dtype, quant_config.is_scale_swizzled
    )
    assert a1q_scale is not None
    return a1q_scale


class MoEPrepareAndFinalizeNaiveDPEPModular(mk.FusedMoEPrepareAndFinalizeModular):
    """Naive Prepare/Finalize for Dp/Ep case for Modular Kernels.

    Uses Torch AR/RS or AR for dispatch/combine operations, applied
    to the topk weights and ids.
    """

    def __init__(
        self,
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
    ) -> None:
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel
        self._num_dispatchers = num_dispatchers
        # Set by FusedMoEWithLoRA.set_mapping() when LoRA is active. When
        # present, prepare() dispatches the per-token LoRA mapping alongside
        # hidden_states and writes the gathered result back to the context so
        # experts can use the per-rank-local mapping.
        self._lora_context = None

    def set_lora_context(self, ctx) -> None:
        self._lora_context = ctx

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

    def output_is_reduced(self) -> bool:
        return False

    def _allocate_combine_input(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        return get_ep_group().allocate_combine_input(
            shape,
            dtype,
            device,
            is_sequence_parallel=self.is_sequence_parallel,
        )

    def allocate_fused_expert_output(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> torch.Tensor | None:
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP):
            return self._allocate_combine_input(shape, dtype, device)
        return None

    def supports_dbo(self) -> bool:
        # prepare()/finalize() hand the collective to the comm stream
        # themselves (see _comm_region), so the synchronous path is DBO-safe.
        # supports_async() stays False: it also selects MK-internal
        # shared-expert overlap. Only validated on ROCm so far.
        return current_platform.is_rocm()

    @contextmanager
    def _comm_region(self) -> Generator[None]:
        """Scope the dispatch/combine collective runs in.

        Without DBO the collective runs on the calling stream. Under DBO it
        runs on the comm stream so the peer ubatch can compute while it is in
        flight, using the same compute/comm handoff as DeepEP-HT. Graph
        capture keeps every collective on the capture stream, so overlap is
        limited to eager steps (prefill).
        """
        if not dbo_enabled() or torch.cuda.is_current_stream_capturing():
            yield
            return
        # Release the CPU to the peer ubatch so it can enqueue compute while
        # this ubatch's collective runs; the compute stream waits on exit.
        dbo_yield_and_switch_from_compute_to_comm()
        try:
            yield
        finally:
            dbo_switch_to_compute_sync()

    def _dispatch_packed_mxfp4(
        self,
        values: torch.Tensor,
        scales: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> mk.PrepareResultType | None:
        """Gather the MXFP4 dispatch's values, scales, topk ids and topk
        weights as one packed buffer (one all-gather instead of four); None
        when it does not apply and the four tensors go separately."""
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.model_executor.layers.fused_moe.prepare_finalize import (
            mxfp4_dispatch_pack as pack,
        )

        if not rocm_aiter_ops.is_fused_moe_fp4_dispatch_pack_enabled():
            return None
        manager = _packed_dispatch_manager()
        # The layout only depends on shapes and dtypes, which every rank
        # shares, so all ranks take the same path.
        layout = pack.mxfp4_dispatch_layout(values, scales, topk_weights, topk_ids)
        if manager is None or layout is None:
            return None
        buffer = pack.pack_mxfp4_dispatch(
            values, scales, topk_weights, topk_ids, layout
        )
        with self._comm_region():
            # Gathered as a BF16 view, which the all-gather backends take.
            gathered = manager.dispatch_packed(
                buffer.view(torch.bfloat16), self.is_sequence_parallel
            )
        a1q, a1q_scale, topk_weights, topk_ids = pack.unpack_mxfp4_dispatch(
            gathered, layout, values.dtype, scales.dtype
        )
        topk_ids = _localize_shared_expert_ids(topk_ids)
        return a1q, a1q_scale, None, topk_ids, topk_weights

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        """Quantize and Dispatch Topk Weights and Topk Ids."""
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        mxfp4_dtypes: tuple[torch.dtype, torch.dtype] | None = None
        if self._lora_context is None and _use_mxfp4_dispatch(
            a1, quant_config, defer_input_quant
        ):
            from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
                quantize_mxfp4_for_dispatch,
            )

            # The quantization is per row, so quantizing each rank's rows
            # before the gather gives the bytes the experts would compute
            # from the gathered rows. The packed values and the scales are
            # gathered as BF16 views, which the all-gather backends take.
            a1q, a1q_scale_orig = quantize_mxfp4_for_dispatch(a1)
            packed = self._dispatch_packed_mxfp4(
                a1q, a1q_scale_orig, topk_weights, topk_ids
            )
            if packed is not None:
                return packed
            mxfp4_dtypes = (a1q.dtype, a1q_scale_orig.dtype)
            a1q = a1q.view(torch.bfloat16)
            scales = [a1q_scale_orig.view(torch.bfloat16)]
        else:
            a1q, scales, a1q_scale_orig = _quantize_and_setup_dispatch(
                a1, quant_config, defer_input_quant
            )

        # When LoRA is active, dispatch the per-token LoRA id along with
        # hidden_states so every rank receives the correct mapping for the
        # tokens it ends up processing. The punica_wrapper stores indices as
        # int64 but the moe_lora_align_block_size kernel expects int32, so
        # pull the pre-cast view from token_mapping_meta.
        lora_ctx = self._lora_context
        local_token_lora_mapping = None
        if lora_ctx is not None:
            local_token_lora_mapping = (
                lora_ctx.punica_wrapper.token_mapping_meta.token_lora_mapping[
                    : a1.shape[0]
                ]
            )

        extra_tensors: list[torch.Tensor] | None = None
        if scales is not None:
            extra_tensors = list(scales)
        if local_token_lora_mapping is not None:
            if extra_tensors is None:
                extra_tensors = []
            extra_tensors.append(local_token_lora_mapping)

        with self._comm_region():
            res = get_ep_group().dispatch(
                a1q,
                topk_weights,
                topk_ids,
                is_sequence_parallel=self.is_sequence_parallel,
                extra_tensors=extra_tensors,
            )

        if extra_tensors is None:
            assert len(res) == 3
            a1q, topk_weights, topk_ids = res
            a1q_scale = a1q_scale_orig
        else:
            assert len(res) == 4
            a1q, topk_weights, topk_ids, gathered_extras = res
            gathered_extras = list(gathered_extras)
            if local_token_lora_mapping is not None:
                dispatched_lora_mapping = gathered_extras.pop()
                assert lora_ctx is not None
                lora_ctx.local_token_lora_mapping = dispatched_lora_mapping
            if mxfp4_dtypes is not None:
                assert len(gathered_extras) == 1
                a1q = a1q.view(mxfp4_dtypes[0])
                a1q_scale = gathered_extras[0].view(mxfp4_dtypes[1])
            elif scales is not None:
                a1q_scale = _unwrap_scale_and_prepare_for_moe(
                    gathered_extras, quant_config
                )
            else:
                a1q_scale = a1q_scale_orig

        topk_ids = _localize_shared_expert_ids(topk_ids)
        return a1q, a1q_scale, None, topk_ids, topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP):
            # allocate_fused_expert_output already selected the combine input.
            # Reuse it directly; the symmetric allocator may return a new view
            # object for the same high-water backing allocation.
            combine_input = fused_expert_output
        else:
            combine_input = self._allocate_combine_input(
                (topk_ids.shape[0], fused_expert_output.shape[-1]),
                fused_expert_output.dtype,
                fused_expert_output.device,
            )
        out = weight_and_reduce_impl.apply(
            output=combine_input,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        with self._comm_region():
            get_ep_group().combine_into_output(
                out,
                output,
                is_sequence_parallel=self.is_sequence_parallel,
            )


class MoEPrepareAndFinalizeNaiveDPEPMonolithic(mk.FusedMoEPrepareAndFinalizeMonolithic):
    """Naive Prepare/Finalize for Dp/Ep case for Modular Kernels.

    Uses Torch AR/RS or AR for dispatch/combine operations, applied
    to the router logits (the MoE kernel runs the router internally).
    """

    def __init__(
        self,
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
    ) -> None:
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel
        self._num_dispatchers = num_dispatchers

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

    def output_is_reduced(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        router_logits: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
        a1_scale: torch.Tensor | None = None,
    ) -> mk.PrepareMonolithicResultType:
        """Quantize and Dispatch Router Logits."""
        assert a1_scale is None, "naive DP/EP dispatch quantizes its own input"
        a1q, scales, a1q_scale_orig = _quantize_and_setup_dispatch(
            a1, quant_config, defer_input_quant
        )

        res = get_ep_group().dispatch_router_logits(
            a1q,
            router_logits,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=scales,
        )

        if scales is None:
            assert len(res) == 2
            a1q, router_logits = res
            a1q_scale = a1q_scale_orig
        else:
            assert len(res) == 3
            a1q, router_logits, scales = res
            a1q_scale = _unwrap_scale_and_prepare_for_moe(scales, quant_config)

        return a1q, a1q_scale, router_logits

    def supports_precomputed_routing(self) -> bool:
        return True

    def prepare_routed(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Quantize and dispatch topk weights and ids instead of logits."""
        a1q, scales, a1q_scale_orig = _quantize_and_setup_dispatch(
            a1, quant_config, defer_input_quant
        )
        res = get_ep_group().dispatch(
            a1q,
            topk_weights,
            topk_ids,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=scales,
        )
        if scales is None:
            assert len(res) == 3
            a1q, topk_weights, topk_ids = res
            topk_ids = _localize_shared_expert_ids(topk_ids)
            return a1q, a1q_scale_orig, topk_weights, topk_ids
        assert len(res) == 4
        a1q, topk_weights, topk_ids, gathered_scales = res
        a1q_scale = _unwrap_scale_and_prepare_for_moe(gathered_scales, quant_config)
        return a1q, a1q_scale, topk_weights, _localize_shared_expert_ids(topk_ids)

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
    ) -> torch.Tensor:
        out = get_ep_group().combine(
            fused_expert_output, is_sequence_parallel=self.is_sequence_parallel
        )
        return out


def make_moe_prepare_and_finalize_naive_dp_ep(
    use_monolithic: bool,
    is_sequence_parallel: bool = False,
    num_dispatchers: int = 1,
) -> MoEPrepareAndFinalizeNaiveDPEPModular | MoEPrepareAndFinalizeNaiveDPEPMonolithic:
    return (
        MoEPrepareAndFinalizeNaiveDPEPMonolithic(
            is_sequence_parallel=is_sequence_parallel,
            num_dispatchers=num_dispatchers,
        )
        if use_monolithic
        else MoEPrepareAndFinalizeNaiveDPEPModular(
            is_sequence_parallel=is_sequence_parallel,
            num_dispatchers=num_dispatchers,
        )
    )
