# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""π0.5 VLA pipeline for vllm-omni.

The pipeline preprocesses each OpenPI observation, runs one flow-matching
model call for a request batch, and returns one action output per request.

π0.5 is stateless across calls (no KV reuse, first-order Markov), so
``session_id`` / ``reset`` from the OpenPI protocol are accepted but ignored.

The post-processing order is load-bearing and matches LeRobot::

    unnormalize → absolute actions → to_cpu

``AbsoluteActionsProcessorStep`` must run *after* unnormalization, because a
relative-action checkpoint's ``norm_stats`` are computed in relative space.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import fields as dataclass_fields
from uuid import uuid4

import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.pi05.config import SUPPORTED_DTYPE_NAMES, Pi05Config
from vllm_omni.diffusion.models.pi05.modeling_pi05 import Pi05ForActionPrediction
from vllm_omni.diffusion.models.pi05.processor_pi05 import Pi05Processor
from vllm_omni.diffusion.models.pi05_pipeline_config import PI05_PIPELINE as PI05_PIPELINE
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

# π0.5 pins the PaliGemma tokenizer (LeRobot hardcodes it too).
DEFAULT_PI05_TOKENIZER = "google/paligemma-3b-pt-224"

# float16 is absent deliberately, not by oversight: nothing here has been
# validated against it. The float32/bfloat16 trade-off is in the deploy config
# and recipes/lerobot/Pi05.md.
SUPPORTED_DTYPES = (torch.float32, torch.bfloat16)

# The two lists guard different entry points — what the checkpoint declares
# versus the dtype actually cast to — so they must not drift apart.
assert {str(dtype).split(".")[-1] for dtype in SUPPORTED_DTYPES} == set(SUPPORTED_DTYPE_NAMES)


def _benchmark_telemetry_enabled() -> bool:
    if os.environ.get("VLLM_OMNI_BENCH_BATCH_TELEMETRY") != "1":
        return False
    distributed = torch.distributed
    return not distributed.is_available() or not distributed.is_initialized() or distributed.get_rank() == 0


def _benchmark_request_id(observation: Mapping) -> str | None:
    value = observation.get("benchmark_request_id")
    if not isinstance(value, str) or not value.startswith("benchmark-"):
        return None
    suffix = value.removeprefix("benchmark-")
    return value if len(suffix) == 32 and all(char in "0123456789abcdef" for char in suffix) else None


def _checkpoint_declared_keys(model_dir: str) -> set[str]:
    """The keys the checkpoint's config.json actually carries."""
    path = os.path.join(model_dir, "config.json")
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return set(json.load(f))


def _comparable(value):
    """Compare yaml and checkpoint values without tripping on list/tuple."""
    return list(value) if isinstance(value, (list, tuple)) else value


def _pi05_post_process(x):
    """Module-level identity post-process (picklable across the orchestrator's
    multiprocess boundary — a local closure is not)."""
    return x


def get_pi05_post_process_func(od_config: OmniDiffusionConfig):
    """π0.5 returns actions directly; post-processing is identity."""
    del od_config
    return _pi05_post_process


_LEROBOT_FLOAT32_IN_BFLOAT16 = (
    "vision_tower",
    "multi_modal_projector",
    "input_layernorm",
    "post_attention_layernorm",
    "model.norm",
)


def _to_bfloat16_for_inference(model: Pi05ForActionPrediction) -> None:
    """Apply LeRobot's mixed-precision inference layout explicitly."""
    inner_model = model.paligemma_with_expert
    for name, param in inner_model.named_parameters():
        target_dtype = (
            torch.float32 if any(selector in name for selector in _LEROBOT_FLOAT32_IN_BFLOAT16) else torch.bfloat16
        )
        param.data = param.data.to(dtype=target_dtype)
    for buffer in inner_model.buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(dtype=torch.bfloat16)

    for name, param in model.named_parameters():
        if not name.startswith("paligemma_with_expert."):
            param.data = param.data.to(dtype=torch.float32)
    for name, buffer in model.named_buffers():
        if not name.startswith("paligemma_with_expert.") and buffer.is_floating_point():
            buffer.data = buffer.data.to(dtype=torch.float32)


def _set_inference_dtype(model: Pi05ForActionPrediction, dtype: torch.dtype) -> None:
    if dtype == torch.float32:
        model.to(dtype=torch.float32)
    elif dtype == torch.bfloat16:
        _to_bfloat16_for_inference(model)
    else:
        raise ValueError(f"Unsupported π0.5 inference dtype: {dtype!r}.")


class Pi05Pipeline(nn.Module):
    """π0.5 VLA pipeline: raw robot obs → continuous action chunk.

    Registered as ``"Pi05Pipeline"`` in the diffusion registry.
    """

    supports_request_batch = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        self.model_dir = self._resolve_model_dir(od_config.model)
        self.config = self._build_config(od_config)

        custom_args = od_config.custom_pipeline_args or {}
        self.tokenizer_source = str(custom_args.get("tokenizer", self._resolve_tokenizer_source()))

        self._torch_dtype = self._resolve_dtype(od_config)
        self._device = self._resolve_device(od_config)

        self.tokenizer = self._load_tokenizer()
        self.model = self._initialize_model()

        self.processor = Pi05Processor(self.config, self.tokenizer, self._device)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_model_dir(model: str | None) -> str | None:
        """Return a local directory for ``model``; download an HF repo id if needed."""
        if not model:
            return None
        if os.path.isdir(model):
            return model
        # Via repo_utils' shared HfApi rather than huggingface_hub directly, so
        # the download carries vLLM-Omni's user agent like every other repo access.
        from vllm_omni.transformers_utils.repo_utils import hf_api

        return hf_api().snapshot_download(
            repo_id=model,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
        )

    def _build_config(self, od_config: OmniDiffusionConfig) -> Pi05Config:
        """Read the config from the checkpoint, then let the deploy yaml override it."""
        checkpoint_config = Pi05Config.from_pretrained(self.model_dir) if self.model_dir else None
        if checkpoint_config is None:
            return Pi05Config.from_model_config(od_config.model_config)
        if not od_config.model_config:
            return checkpoint_config

        resolved = {
            item.name: getattr(checkpoint_config, item.name) for item in dataclass_fields(Pi05Config) if item.init
        }
        # Only a key the checkpoint actually declares can disagree with the yaml.
        # Comparing against the dataclass default instead would report every
        # serving-only field, such as policy_server_config, on every start.
        declared_keys = _checkpoint_declared_keys(self.model_dir)
        for key, value in od_config.model_config.items():
            if key in resolved and key in declared_keys and _comparable(value) != _comparable(resolved[key]):
                logger.warning(
                    "Pi05Pipeline: the deploy config sets %s=%r, overriding %r from the checkpoint.",
                    key,
                    value,
                    resolved[key],
                )
        resolved.update(od_config.model_config)
        return Pi05Config.from_model_config(resolved)

    def _resolve_tokenizer_source(self) -> str:
        """Prefer the checkpoint dir if it ships tokenizer files; else PaliGemma."""
        if self.model_dir and os.path.isdir(self.model_dir):
            if os.path.exists(os.path.join(self.model_dir, "tokenizer_config.json")):
                return self.model_dir
        return DEFAULT_PI05_TOKENIZER

    @staticmethod
    def _resolve_dtype(od_config: OmniDiffusionConfig) -> torch.dtype:
        """Resolve the dtype the weights are actually cast to.

        This is the load-bearing check, not ``Pi05Config.dtype``: the cast in
        :meth:`_initialize_model` reads the *top-level* ``OmniDiffusionConfig``
        field, so a guard on the model config alone would let an unsupported
        dtype through. See :data:`SUPPORTED_DTYPES` for why float16 is excluded.
        """
        dt = od_config.dtype
        resolved = dt if isinstance(dt, torch.dtype) else getattr(torch, str(dt).split(".")[-1], None)
        if resolved not in SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported π0.5 dtype: {dt!r}. Supported: "
                f"{', '.join(sorted(str(d).split('.')[-1] for d in SUPPORTED_DTYPES))}."
            )
        return resolved

    @staticmethod
    def _resolve_device(od_config: OmniDiffusionConfig) -> torch.device:
        from vllm_omni.diffusion.distributed.utils import get_local_device

        try:
            return get_local_device()
        except Exception:  # noqa: BLE001
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        # padding_side="right" is part of the π0.5 spec: the prefix is a fixed
        # 200-token block whose live tokens must start at index 0.
        return AutoTokenizer.from_pretrained(self.tokenizer_source, padding_side="right")

    def has_real_checkpoint(self) -> bool:
        return bool(self.model_dir) and os.path.exists(os.path.join(self.model_dir, "model.safetensors"))

    def _initialize_model(self) -> Pi05ForActionPrediction:
        if not self.has_real_checkpoint():
            expected = os.path.join(self.model_dir or "<missing-model-dir>", "model.safetensors")
            raise FileNotFoundError(f"π0.5 serving requires checkpoint weights at {expected}.")
        model = Pi05ForActionPrediction(self.config)
        _set_inference_dtype(model, self._torch_dtype)
        model.to(device=self._device)
        self._load_checkpoint(model)
        model.eval()
        return model

    def _load_checkpoint(self, model: Pi05ForActionPrediction) -> None:
        import safetensors.torch

        path = os.path.join(self.model_dir, "model.safetensors")
        logger.info("Pi05Pipeline: loading π0.5 weights from %s", path)
        try:
            state = safetensors.torch.load_file(path)
            model.load_weights(state.items())
        except Exception as exc:
            raise RuntimeError(f"Failed to load complete π0.5 checkpoint {path}: {exc}") from exc

    # ------------------------------------------------------------------
    # Framework weight-loading hook
    # ------------------------------------------------------------------
    def load_weights(self, weights=()):  # noqa: D401
        """No-op for the diffusion loader: π0.5 self-loads its checkpoint."""
        for _ in weights:
            pass
        return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch, **kwargs) -> list[DiffusionOutput]:
        del kwargs
        outputs: list[DiffusionOutput | None] = [None] * req.num_reqs
        valid_indices: list[int] = []
        observations: list[Mapping] = []
        inputs: list[tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]] = []
        generators: list[torch.Generator | None] = []
        num_steps: int | None = None

        for index, request in enumerate(req.requests):
            sampling = request.sampling_params
            robot_obs = (sampling.extra_args or {}).get("robot_obs")
            if robot_obs is None:
                prompt = request.prompt
                if isinstance(prompt, str):
                    prompt_text = prompt
                elif isinstance(prompt, Mapping):
                    prompt_text = prompt.get("prompt", "")
                else:
                    prompt_text = ""
                if prompt_text == "dummy run" or sampling.num_inference_steps == 1:
                    outputs[index] = DiffusionOutput(
                        output={"actions": np.zeros((self.config.chunk_size, self.config.action_dim), np.float32)}
                    )
                else:
                    outputs[index] = DiffusionOutput(
                        error="Pi05Pipeline.forward requires sampling_params.extra_args['robot_obs']."
                    )
                continue

            try:
                if not isinstance(robot_obs, Mapping):
                    raise TypeError("robot_obs must be a mapping")
                steps = sampling.num_inference_steps
                if steps is not None and (
                    isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or int(steps) < 1
                ):
                    raise ValueError(f"num_inference_steps must be a positive integer, got {steps!r}.")
                if valid_indices and (int(steps) if steps is not None else None) != num_steps:
                    raise ValueError("num_inference_steps differs from the current request batch.")
                model_inputs = self.processor.build_model_inputs(robot_obs)
                generator = sampling.generator
                if isinstance(generator, list):
                    if len(generator) != 1:
                        raise ValueError("π0.5 expects one generator per request.")
                    generator = generator[0]
                if generator is None and getattr(sampling, "seed", None) is not None:
                    generator = torch.Generator(device=self._device).manual_seed(int(sampling.seed))
                if generator is not None and not isinstance(generator, torch.Generator):
                    raise TypeError("π0.5 request generator must be a torch.Generator.")
            except Exception as exc:  # A malformed observation fails only its own request.
                outputs[index] = DiffusionOutput.from_exception(exc)
                continue

            valid_indices.append(index)
            observations.append(robot_obs)
            inputs.append(model_inputs)
            generators.append(generator)
            num_steps = int(steps) if steps is not None else None

        if valid_indices:
            images = [
                torch.cat([item[0][camera] for item in inputs], dim=0)
                for camera in range(self.config.max_cameras)
            ]
            image_masks = [
                torch.cat([item[1][camera] for item in inputs], dim=0) for camera in range(self.config.max_cameras)
            ]
            lang_tokens = torch.cat([item[2] for item in inputs], dim=0)
            lang_masks = torch.cat([item[3] for item in inputs], dim=0)
            model_generator = generators[0] if len(generators) == 1 else generators

            telemetry_enabled = _benchmark_telemetry_enabled()
            start_event = end_event = None
            if telemetry_enabled and self._device.type == "cuda":
                try:
                    device_module = torch.get_device_module(self._device)
                    start_event = device_module.Event(enable_timing=True)
                    end_event = device_module.Event(enable_timing=True)
                    start_event.record()
                except Exception:
                    # Benchmark instrumentation must never change inference.
                    start_event = end_event = None
                    logger.warning("Pi05 benchmark CUDA timing unavailable", exc_info=True)
            forward_start_ns = time.monotonic_ns() if telemetry_enabled else 0
            model_error: BaseException | None = None
            try:
                actions = self.model.sample_actions(
                    images=images,
                    image_masks=image_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    num_steps=num_steps,
                    generator=model_generator,
                )
            except BaseException as exc:
                model_error = exc
                raise
            finally:
                if telemetry_enabled:
                    forward_end_ns = time.monotonic_ns()
                    try:
                        cuda_event_ms = None
                        cuda_sync_overhead_ms = None
                        if start_event is not None and end_event is not None:
                            end_event.record()
                            sync_start_ns = time.monotonic_ns()
                            end_event.synchronize()
                            cuda_event_ms = start_event.elapsed_time(end_event)
                            cuda_sync_overhead_ms = (time.monotonic_ns() - sync_start_ns) / 1e6
                        peak_reserved_mb = (
                            current_omni_platform.max_memory_reserved() / (1024**2)
                            if self._device.type == "cuda" else None
                        )
                    except Exception:
                        cuda_event_ms = cuda_sync_overhead_ms = peak_reserved_mb = None
                        logger.warning("Pi05 benchmark device metrics unavailable", exc_info=True)
                    try:
                        od_config = getattr(self, "od_config", None)
                        event = {
                            "schema": "vllm_omni.pi05.sample_actions.v1",
                            "batch_id": uuid4().hex,
                            "benchmark_request_ids": [
                                _benchmark_request_id(observation) for observation in observations
                            ],
                            "engine_request_ids": [req.requests[index].request_id for index in valid_indices],
                            "session_ids": [observation.get("session_id") for observation in observations],
                            "executed_batch_size": len(valid_indices),
                            "configured_max_num_seqs": getattr(od_config, "max_num_seqs", None),
                            "configured_request_batch_max_wait_ms": getattr(
                                od_config, "request_batch_max_wait_ms", None
                            ),
                            "scheduler_queue_wait_ms": [
                                req.requests[index].scheduler_queue_wait_ms for index in valid_indices
                            ],
                            "forward_start_ns": forward_start_ns,
                            "forward_end_ns": forward_end_ns,
                            "cpu_forward_ms": (forward_end_ns - forward_start_ns) / 1e6,
                            "cuda_event_ms": cuda_event_ms,
                            "cuda_sync_overhead_ms": cuda_sync_overhead_ms,
                            "peak_reserved_mb": peak_reserved_mb,
                            "status": "completed" if model_error is None else "failed",
                            "error_type": type(model_error).__name__ if model_error is not None else None,
                        }
                        logger.info("BENCH_PI05_FORWARD %s", json.dumps(event, separators=(",", ":")))
                    except Exception:
                        logger.warning("Pi05 benchmark telemetry emission failed", exc_info=True)
            expected_shape = (len(valid_indices), self.config.chunk_size, self.config.max_action_dim)
            if tuple(actions.shape) != expected_shape:
                raise RuntimeError(
                    f"π0.5 model returned actions with shape {tuple(actions.shape)}, expected {expected_shape}."
                )
            for row, index in enumerate(valid_indices):
                try:
                    action = self.processor.build_model_outputs(actions[row : row + 1], observations[row])
                    outputs[index] = DiffusionOutput(output={"actions": action})
                except Exception as exc:  # Postprocessing uses this request's state.
                    outputs[index] = DiffusionOutput.from_exception(exc)

        assert all(output is not None for output in outputs)
        return [output for output in outputs if output is not None]
