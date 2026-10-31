# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import os
import time
from collections.abc import Iterable, Mapping
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.benchmark_expected_cohort import forward_cohort_telemetry
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.gr00t.policy import Gr00tPolicy
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

logger = init_logger(__name__)


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


def _to_float32_action_dict(actions: Mapping[str, Any]) -> dict[str, np.ndarray]:
    converted = {str(key): np.asarray(value, dtype=np.float32) for key, value in actions.items()}
    if not converted:
        raise RuntimeError("GR00T policy returned an empty action dict.")
    return converted


class Gr00tN1d7Pipeline(nn.Module):
    """GR00T N1.7 policy pipeline backed by vLLM-Omni's local GR00T port.

    vLLM-Omni owns the serving integration: OpenPI observations arrive through
    `sampling_params.extra_args["robot_obs"]`, this pipeline runs GR00T policy
    inference, and actions are returned through `DiffusionOutput.output["actions"]`.

    The request's `seed` (materialised by the runner into `sampling_params.generator`)
    draws the flow-matching noise, so repeating a request with the same seed reproduces
    its action chunk.
    """

    supports_request_batch = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        model_config = od_config.model_config
        self.model_path = od_config.model
        self.od_config = od_config
        self.embodiment_tag = str(model_config.get("embodiment_tag") or "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT")
        self.strict = bool(model_config.get("strict", True))
        self.device = get_local_device()

        logger.info("Loading GR00T N1.7 policy from %s with embodiment_tag=%s", self.model_path, self.embodiment_tag)
        self.policy = Gr00tPolicy(
            model_path=self.model_path,
            embodiment_tag=self.embodiment_tag,
            device=self.device,
            strict=self.strict,
        )
        self._validate_policy_server_config(model_config.get("policy_server_config"))

    def _validate_policy_server_config(self, psc: Mapping[str, Any] | None) -> None:
        """Fail fast if the deploy handshake drifts from the loaded checkpoint.

        ``policy_server_config`` is sent verbatim to the OpenPI client, so its
        model/embodiment-specific values must match what the loaded policy actually
        produces; otherwise the client is handed the wrong action contract.
        """
        if not isinstance(psc, Mapping):
            return
        action_config = self.policy.modality_configs["action"]
        expected_horizon = len(action_config.delta_indices)
        expected_keys = set(action_config.modality_keys)

        if "action_horizon" in psc and psc["action_horizon"] != expected_horizon:
            raise ValueError(
                f"policy_server_config.action_horizon={psc['action_horizon']} != loaded model's "
                f"action horizon {expected_horizon}."
            )
        if "action_keys" in psc and set(psc["action_keys"]) != expected_keys:
            raise ValueError(
                f"policy_server_config.action_keys={list(psc['action_keys'])} != loaded model's "
                f"action keys {sorted(expected_keys)}."
            )
        psc_embodiment = psc.get("embodiment_tag")
        if psc_embodiment is not None and psc_embodiment != self.embodiment_tag:
            raise ValueError(
                f"policy_server_config.embodiment_tag={psc_embodiment!r} != "
                f"model_config.embodiment_tag={self.embodiment_tag!r}."
            )

    def reset(self) -> dict[str, Any]:
        return self.policy.reset() or {}

    @property
    def weights_sources(self) -> tuple[Any, ...]:
        return ()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        consumed = list(weights)
        if consumed:
            raise RuntimeError(
                f"Gr00tN1d7Pipeline.load_weights received {len(consumed)} weight tensors; "
                "weights_sources=() should prevent this. GR00T weights are loaded directly by Gr00tPolicy."
            )
        return set()

    def _dummy_actions(self) -> dict[str, np.ndarray]:
        embodiment_value = self.policy.embodiment_tag.value
        action_config = self.policy.modality_configs["action"]
        horizon = len(action_config.delta_indices)
        norm_params = self.policy.processor.state_action_processor.norm_params[embodiment_value]["action"]
        actions = {}
        for key in action_config.modality_keys:
            dim = norm_params[key]["dim"]
            dim = int(dim.item() if hasattr(dim, "item") else dim)
            actions[key] = np.zeros((1, horizon, dim), dtype=np.float32)
        return actions

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch, **kwargs) -> list[DiffusionOutput]:
        del kwargs
        outputs: list[DiffusionOutput | None] = [None] * req.num_reqs
        valid_indices: list[int] = []
        observations: list[dict[str, Any]] = []
        wire_observations: list[Mapping[str, Any]] = []
        generators: list[torch.Generator | None] = []
        signature: tuple[Any, ...] | None = None

        for index, request in enumerate(req.requests):
            sampling = request.sampling_params
            extra_args = sampling.extra_args or {}
            robot_obs = extra_args.get("robot_obs")
            if robot_obs is None:
                if request.request_id == DUMMY_DIFFUSION_REQUEST_ID:
                    outputs[index] = DiffusionOutput(output={"actions": self._dummy_actions()})
                else:
                    outputs[index] = DiffusionOutput(
                        error="Gr00tN1d7Pipeline.forward expects sampling_params.extra_args['robot_obs']."
                    )
                continue

            try:
                if not isinstance(robot_obs, Mapping):
                    raise TypeError(f"robot_obs must be a dict, got {type(robot_obs).__name__}.")
                policy_obs = _normalize_observation(robot_obs, language_key=self.policy.language_key)
                row_signature = _single_observation_signature(policy_obs)
                if self.strict:
                    self.policy.check_observation(policy_obs)
                if signature is not None and row_signature != signature:
                    raise ValueError("GR00T observation modalities or shapes differ from the current batch.")
                generator = _noise_generator(sampling, self.device)
            except Exception as exc:  # A malformed observation fails only its own request.
                outputs[index] = DiffusionOutput.from_exception(exc)
                continue

            if extra_args.get("reset"):
                self.reset()
            signature = row_signature
            valid_indices.append(index)
            observations.append(policy_obs)
            wire_observations.append(robot_obs)
            generators.append(generator)

        if valid_indices:
            policy_obs = _collate_observations(observations)
            if len(generators) > 1:
                options = {"generators": generators}
            elif generators[0] is not None:
                options = {"generator": generators[0]}
            else:
                options = None
            telemetry_enabled = _benchmark_telemetry_enabled()
            start_event = end_event = None
            if telemetry_enabled and self.device.type == "cuda":
                try:
                    device_module = torch.get_device_module(self.device)
                    start_event = device_module.Event(enable_timing=True)
                    end_event = device_module.Event(enable_timing=True)
                    start_event.record()
                except Exception:
                    start_event = end_event = None
                    logger.warning("GR00T benchmark CUDA timing unavailable", exc_info=True)
            forward_start_ns = time.monotonic_ns() if telemetry_enabled else 0
            model_error: BaseException | None = None
            try:
                result = self.policy.get_action(policy_obs, options)
            except BaseException as exc:
                model_error = exc
                raise
            finally:
                if telemetry_enabled:
                    forward_end_ns = time.monotonic_ns()
                    cuda_event_ms = cuda_sync_overhead_ms = None
                    try:
                        if start_event is not None and end_event is not None:
                            end_event.record()
                            sync_start_ns = time.monotonic_ns()
                            end_event.synchronize()
                            cuda_event_ms = start_event.elapsed_time(end_event)
                            cuda_sync_overhead_ms = (time.monotonic_ns() - sync_start_ns) / 1e6
                    except Exception:
                        logger.warning("GR00T benchmark CUDA timing unavailable", exc_info=True)
                    try:
                        event = {
                            "schema": "vllm_omni.gr00t.get_action.v1",
                            "batch_id": uuid4().hex,
                            "benchmark_request_ids": [
                                _benchmark_request_id(observation) for observation in wire_observations
                            ],
                            "engine_request_ids": [req.requests[index].request_id for index in valid_indices],
                            "session_ids": [observation.get("session_id") for observation in wire_observations],
                            "executed_batch_size": len(valid_indices),
                            "configured_max_num_seqs": getattr(self.od_config, "max_num_seqs", None),
                            "configured_request_batch_max_wait_ms": getattr(
                                self.od_config, "request_batch_max_wait_ms", None
                            ),
                            "scheduler_queue_wait_ms": [
                                req.requests[index].scheduler_queue_wait_ms for index in valid_indices
                            ],
                            "forward_start_ns": forward_start_ns,
                            "forward_end_ns": forward_end_ns,
                            "cpu_forward_ms": (forward_end_ns - forward_start_ns) / 1e6,
                            "cuda_event_ms": cuda_event_ms,
                            "cuda_sync_overhead_ms": cuda_sync_overhead_ms,
                            "status": "completed" if model_error is None else "failed",
                            "error_type": type(model_error).__name__ if model_error is not None else None,
                        }
                        event.update(forward_cohort_telemetry(wire_observations))
                        if os.environ.get("VLLM_OMNI_BENCH_DETAILED_TRACE") == "1":
                            traced = wire_observations
                            event.update(
                                trace_schema_version="control_loop_trace.v1",
                                clock_domain="host_monotonic_ns",
                                benchmark_run_ids=[obs.get("benchmark_run_id") for obs in traced],
                                server_received_ns=[obs.get("benchmark_server_received_ns") for obs in traced],
                                server_queue_enter_ns=[obs.get("benchmark_server_queue_enter_ns") for obs in traced],
                                server_queue_leave_ns=[obs.get("benchmark_server_queue_leave_ns") for obs in traced],
                            )
                            opened = [obs.get("benchmark_batch_opened_ns") for obs in traced]
                            closed = [obs.get("benchmark_batch_closed_ns") for obs in traced]
                            if opened and all(value == opened[0] for value in opened) and \
                                    all(value == closed[0] for value in closed) and \
                                    opened[0] is not None and closed[0] is not None:
                                event["batch_opened_ns"] = opened[0]
                                event["batch_closed_ns"] = closed[0]
                                event["actual_batch_wait_ms"] = (closed[0] - opened[0]) / 1e6
                        os.write(2, ("BENCH_GR00T_FORWARD " + json.dumps(event, separators=(",", ":")) + "\n").encode())
                    except Exception:
                        logger.warning("GR00T benchmark telemetry emission failed", exc_info=True)
            actions = result[0] if isinstance(result, tuple) else result
            if not isinstance(actions, Mapping):
                raise TypeError(f"GR00T policy returned {type(actions).__name__}; expected dict actions.")
            converted = _to_float32_action_dict(actions)
            for key, value in converted.items():
                if value.ndim != 3 or value.shape[0] != len(valid_indices):
                    raise ValueError(
                        f"GR00T action {key!r} has shape {value.shape}; expected one row per valid request."
                    )
            for row, index in enumerate(valid_indices):
                outputs[index] = DiffusionOutput(
                    output={"actions": {key: value[row : row + 1] for key, value in converted.items()}}
                )

        assert all(output is not None for output in outputs)
        return [output for output in outputs if output is not None]


def _single_observation_signature(observation: Mapping[str, Any]) -> tuple[Any, ...]:
    """Check one GR00T wire observation and describe its batchable structure."""
    signature: list[Any] = []
    for modality, ndim in (("video", 5), ("state", 3)):
        values = observation.get(modality)
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"GR00T observation requires nonempty {modality!r} data.")
        for key, value in sorted(values.items()):
            if not isinstance(value, np.ndarray) or value.ndim != ndim or value.shape[0] != 1:
                raise ValueError(f"GR00T {modality}.{key} must have one leading batch row and {ndim} dimensions.")
            signature.append((modality, key, value.shape[1:], value.dtype.str))
    language = observation.get("language")
    if not isinstance(language, Mapping) or not language:
        raise ValueError("GR00T observation requires language data.")
    for key, value in sorted(language.items()):
        if not isinstance(value, list) or len(value) != 1:
            raise ValueError(f"GR00T language.{key} must contain one outer batch row.")
        if not isinstance(value[0], list) or not value[0] or any(not isinstance(item, str) for item in value[0]):
            raise ValueError(f"GR00T language.{key} must contain a nonempty list of strings.")
        signature.append(("language", key))
    return tuple(signature)


def _collate_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    first = observations[0]
    return {
        "video": {key: np.concatenate([obs["video"][key] for obs in observations], axis=0) for key in first["video"]},
        "state": {key: np.concatenate([obs["state"][key] for obs in observations], axis=0) for key in first["state"]},
        "language": {
            key: [row for obs in observations for row in obs["language"][key]]
            for key in first["language"]
        },
    }


def _noise_generator(
    sampling_params: OmniDiffusionSamplingParams, device: torch.device | str
) -> torch.Generator | None:
    """Pick the generator that draws the action head's initial noise.

    The runner materialises ``sampling_params.seed`` into ``sampling_params.generator`` on
    the model device before every forward, and ``OmniDiffusionRequest`` assigns a random
    seed when the caller omits one, so in serving the generator is always present. The
    seed fallback serves callers that bypass the runner (direct use, unit tests). ``None``
    leaves the action head on the global RNG, as upstream Isaac-GR00T runs. GR00T draws one
    noise tensor per request, so a generator in any other form (for example a list of several
    generators) is ignored with a warning and the seed fallback applies.
    """
    generator = sampling_params.generator
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]
    if isinstance(generator, torch.Generator):
        return generator
    if generator is not None:
        received = (
            f"a list of {len(generator)} generators" if isinstance(generator, list) else type(generator).__name__
        )
        logger.warning(
            "GR00T expects a single torch.Generator per request but got %s; ignoring it. The initial noise "
            "falls back to sampling_params.seed=%s, or to the global RNG when that is None.",
            received,
            sampling_params.seed,
        )
    if sampling_params.seed is not None:
        return torch.Generator(device=device).manual_seed(int(sampling_params.seed))
    return None


def _normalize_observation(robot_obs: Mapping[str, Any], *, language_key: str) -> dict[str, Any]:
    obs: dict[str, Any] = {}
    if "video" in robot_obs:
        obs["video"] = robot_obs["video"]
    elif "images" in robot_obs:
        obs["video"] = robot_obs["images"]
    if "state" in robot_obs:
        obs["state"] = robot_obs["state"]
    if "language" in robot_obs:
        obs["language"] = robot_obs["language"]
    else:
        prompt = robot_obs.get("prompt")
        if prompt is not None:
            obs["language"] = {language_key: [[str(prompt)]]}
    return obs
