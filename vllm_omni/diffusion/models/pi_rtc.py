# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local guided Real-Time Chunking for the Pi policy family."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
import torch

RTC_SCHEMA_VERSION = 1
RTC_MODE = "guided"
RTC_EXTRA_ARG = "rtc"

RTCSchedule = Literal["linear", "exp", "ones", "zeros"]
_RTC_SCHEDULES = frozenset(("linear", "exp", "ones", "zeros"))
_RTC_KEYS = frozenset(
    (
        "schema_version",
        "mode",
        "previous_actions",
        "inference_delay",
        "execution_horizon",
        "prefix_attention_schedule",
        "max_guidance_weight",
        "correlation",
    )
)


@dataclass(frozen=True)
class GuidedRTCRequest:
    """Validated RTC controls for one request."""

    previous_actions: torch.Tensor
    inference_delay: int
    execution_horizon: int
    prefix_attention_schedule: RTCSchedule
    max_guidance_weight: float
    correlation: dict[str, str | int | float | bool | None]

    def response_metadata(self) -> RTCResponseMetadata:
        return {
            "schema_version": RTC_SCHEMA_VERSION,
            "mode": RTC_MODE,
            "correlation": self.correlation,
        }


@dataclass(frozen=True)
class GuidedRTCBatch:
    """Device tensors needed to guide a heterogeneous request batch."""

    previous_actions: torch.Tensor
    prefix_weights: torch.Tensor
    max_guidance_weights: torch.Tensor


class RTCResponseMetadata(TypedDict):
    schema_version: int
    mode: str
    correlation: dict[str, str | int | float | bool | None]


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"rtc.{name} must be an integer, got {value!r}.")
    return int(value)


def parse_guided_rtc_request(
    raw: object,
    *,
    action_horizon: int,
    action_dim: int,
    device: torch.device,
) -> GuidedRTCRequest | None:
    """Validate a versioned RTC request and materialize its action prefix."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("sampling_params.rtc must be a mapping.")
    if not all(isinstance(key, str) for key in raw):
        raise TypeError("sampling_params.rtc keys must be strings.")

    unknown = set(raw) - _RTC_KEYS
    if unknown:
        raise ValueError(f"sampling_params.rtc has unknown fields: {sorted(unknown)}.")
    version = _require_int(raw.get("schema_version"), "schema_version")
    if version != RTC_SCHEMA_VERSION:
        raise ValueError(f"rtc.schema_version must be {RTC_SCHEMA_VERSION}, got {version}.")
    mode = raw.get("mode")
    if mode != RTC_MODE:
        raise ValueError(f"rtc.mode must be {RTC_MODE!r}, got {mode!r}.")

    delay = _require_int(raw.get("inference_delay"), "inference_delay")
    horizon = _require_int(raw.get("execution_horizon"), "execution_horizon")
    if not 0 <= delay <= horizon <= action_horizon:
        raise ValueError(
            "rtc timing must satisfy "
            f"0 <= inference_delay <= execution_horizon <= {action_horizon}; got {delay}, {horizon}."
        )

    schedule = raw.get("prefix_attention_schedule", "linear")
    if schedule not in _RTC_SCHEDULES:
        raise ValueError(f"rtc.prefix_attention_schedule must be one of {sorted(_RTC_SCHEDULES)}, got {schedule!r}.")
    raw_max_guidance_weight = raw.get("max_guidance_weight", 10.0)
    if isinstance(raw_max_guidance_weight, bool) or not isinstance(
        raw_max_guidance_weight, int | float | np.integer | np.floating
    ):
        raise TypeError("rtc.max_guidance_weight must be numeric.")
    max_guidance_weight = float(raw_max_guidance_weight)
    if not np.isfinite(max_guidance_weight) or max_guidance_weight < 0:
        raise ValueError("rtc.max_guidance_weight must be finite and non-negative.")

    if "previous_actions" not in raw:
        raise ValueError("rtc.previous_actions is required.")
    previous_actions = torch.as_tensor(raw["previous_actions"], dtype=torch.float32, device=device)
    expected = (action_horizon, action_dim)
    if tuple(previous_actions.shape) != expected:
        raise ValueError(f"rtc.previous_actions must have shape {expected}, got {tuple(previous_actions.shape)}.")
    if not torch.isfinite(previous_actions).all().item():
        raise ValueError("rtc.previous_actions must contain only finite values.")

    raw_correlation = raw.get("correlation", {})
    if not isinstance(raw_correlation, Mapping):
        raise TypeError("rtc.correlation must be a mapping.")
    correlation: dict[str, str | int | float | bool | None] = {}
    for key, value in raw_correlation.items():
        if not isinstance(key, str) or not isinstance(value, str | int | float | bool | type(None)):
            raise TypeError("rtc.correlation must contain string keys and scalar values.")
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("rtc.correlation float values must be finite.")
        correlation[key] = value

    return GuidedRTCRequest(
        previous_actions=previous_actions,
        inference_delay=delay,
        execution_horizon=horizon,
        prefix_attention_schedule=schedule,
        max_guidance_weight=max_guidance_weight,
        correlation=correlation,
    )


def rtc_prefix_weights(
    action_horizon: int,
    inference_delay: int,
    execution_horizon: int,
    schedule: RTCSchedule,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Build Pi RTC prefix-attention weights for one action chunk."""
    weights = torch.zeros(action_horizon, dtype=torch.float32, device=device)
    if execution_horizon == 0:
        return weights
    if schedule == "zeros":
        weights[:inference_delay] = 1.0
        return weights
    if schedule == "ones":
        weights[:execution_horizon] = 1.0
        return weights
    weights[:inference_delay] = 1.0
    ramp_length = execution_horizon - inference_delay
    if ramp_length:
        ramp = torch.linspace(1.0, 0.0, ramp_length + 2, device=device)[1:-1]
        if schedule == "exp":
            ramp = ramp * torch.expm1(ramp) / (np.e - 1)
        weights[inference_delay:execution_horizon] = ramp
    return weights


def build_guided_rtc_batch(
    requests: Sequence[GuidedRTCRequest | None],
    *,
    action_horizon: int,
    action_dim: int,
    device: torch.device,
) -> GuidedRTCBatch | None:
    """Collate RTC controls while leaving non-RTC rows mathematically unchanged."""
    if not any(request is not None for request in requests):
        return None
    previous_rows: list[torch.Tensor] = []
    weight_rows: list[torch.Tensor] = []
    guidance_weights: list[float] = []
    for request in requests:
        if request is None:
            previous_rows.append(torch.zeros((action_horizon, action_dim), device=device))
            weight_rows.append(torch.zeros(action_horizon, device=device))
            guidance_weights.append(0.0)
            continue
        previous_rows.append(request.previous_actions)
        weight_rows.append(
            rtc_prefix_weights(
                action_horizon,
                request.inference_delay,
                request.execution_horizon,
                request.prefix_attention_schedule,
                device=device,
            )
        )
        guidance_weights.append(request.max_guidance_weight)
    return GuidedRTCBatch(
        previous_actions=torch.stack(previous_rows),
        prefix_weights=torch.stack(weight_rows).unsqueeze(-1),
        max_guidance_weights=torch.tensor(guidance_weights, dtype=torch.float32, device=device),
    )


def guided_rtc_velocity(
    x_t: torch.Tensor,
    time: torch.Tensor,
    denoiser: Callable[[torch.Tensor], torch.Tensor],
    rtc: GuidedRTCBatch,
) -> torch.Tensor:
    """Apply Pi's VJP-based RTC correction to a denoising velocity."""
    with torch.enable_grad():
        differentiable_x = x_t.detach().requires_grad_(True)
        velocity = denoiser(differentiable_x)
        time_bc = time.to(dtype=differentiable_x.dtype)[:, None, None]
        clean_actions = differentiable_x - time_bc * velocity
        error = ((rtc.previous_actions.to(dtype=clean_actions.dtype) - clean_actions) * rtc.prefix_weights).detach()
        correction = torch.autograd.grad(
            clean_actions,
            differentiable_x,
            grad_outputs=error,
            create_graph=False,
        )[0]

        tau = 1.0 - time
        squared = time.square()
        inverse_r_squared = (squared + tau.square()) / squared
        coefficient = time / tau
        unbounded = torch.nan_to_num(
            coefficient * inverse_r_squared,
            nan=0.0,
            posinf=torch.finfo(time.dtype).max,
        )
        guidance = torch.minimum(unbounded, rtc.max_guidance_weights).to(velocity.dtype)
        return (velocity - guidance[:, None, None] * correction).detach()
