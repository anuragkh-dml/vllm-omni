# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.pi_rtc import (
    GuidedRTCBatch,
    build_guided_rtc_batch,
    guided_rtc_velocity,
    parse_guided_rtc_request,
    rtc_prefix_weights,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(**overrides):
    request = {
        "schema_version": 1,
        "mode": "guided",
        "previous_actions": np.ones((4, 2), dtype=np.float32),
        "inference_delay": 1,
        "execution_horizon": 3,
        "correlation": {"generation": 7},
    }
    request.update(overrides)
    return request


def test_parse_guided_rtc_request_is_strict_and_opt_in():
    assert parse_guided_rtc_request(None, action_horizon=4, action_dim=2, device=torch.device("cpu")) is None

    parsed = parse_guided_rtc_request(
        _request(),
        action_horizon=4,
        action_dim=2,
        device=torch.device("cpu"),
    )

    assert parsed is not None
    assert parsed.prefix_attention_schedule == "linear"
    assert parsed.max_guidance_weight == 10.0
    assert parsed.response_metadata()["correlation"] == {"generation": 7}


@pytest.mark.parametrize(
    "override,match",
    [
        ({"schema_version": 2}, "schema_version"),
        ({"mode": "replacement"}, "mode"),
        ({"inference_delay": 4}, "timing"),
        ({"previous_actions": np.zeros((3, 2))}, "shape"),
        ({"max_guidance_weight": np.inf}, "finite"),
        ({"max_guidance_weight": "10"}, "numeric"),
        ({"unexpected": True}, "unknown fields"),
    ],
)
def test_parse_guided_rtc_request_rejects_malformed_controls(override, match):
    with pytest.raises((TypeError, ValueError), match=match):
        parse_guided_rtc_request(
            _request(**override),
            action_horizon=4,
            action_dim=2,
            device=torch.device("cpu"),
        )


def test_prefix_attention_schedules_match_the_rtc_contract():
    device = torch.device("cpu")

    linear = rtc_prefix_weights(5, 1, 4, "linear", device=device)
    ones = rtc_prefix_weights(5, 1, 4, "ones", device=device)
    zeros = rtc_prefix_weights(5, 1, 4, "zeros", device=device)

    torch.testing.assert_close(linear, torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))
    torch.testing.assert_close(ones, torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0]))
    torch.testing.assert_close(zeros, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))


def test_guided_velocity_includes_the_clean_action_vjp():
    x_t = torch.tensor([[[2.0]]])
    time = torch.tensor([0.5])
    rtc = GuidedRTCBatch(
        previous_actions=torch.tensor([[[3.0]]]),
        prefix_weights=torch.ones((1, 1, 1)),
        max_guidance_weights=torch.tensor([10.0]),
    )

    actual = guided_rtc_velocity(x_t, time, lambda actions: 0.5 * actions, rtc)

    # clean=0.75*x=1.5, error=1.5, VJP=0.75*error=1.125,
    # and the t=0.5 guidance weight is 2.
    torch.testing.assert_close(actual, torch.tensor([[[-1.25]]]))


def test_mixed_batch_leaves_non_rtc_row_unchanged():
    device = torch.device("cpu")
    parsed = parse_guided_rtc_request(_request(), action_horizon=4, action_dim=2, device=device)
    rtc = build_guided_rtc_batch(
        [parsed, None],
        action_horizon=4,
        action_dim=2,
        device=device,
    )
    assert rtc is not None
    x_t = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    baseline = 0.25 * x_t

    actual = guided_rtc_velocity(x_t, torch.tensor([0.5, 0.5]), lambda actions: 0.25 * actions, rtc)

    torch.testing.assert_close(actual[1], baseline[1], rtol=0, atol=0)
