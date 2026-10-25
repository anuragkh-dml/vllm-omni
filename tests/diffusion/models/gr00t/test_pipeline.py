# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.gr00t import pipeline_gr00t
from vllm_omni.diffusion.models.gr00t.pipeline_gr00t import Gr00tN1d7Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeGr00tPolicy:
    instances: list["FakeGr00tPolicy"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.reset_calls = 0
        self.seen_obs: dict[str, Any] | None = None
        self.embodiment_tag = SimpleNamespace(value="fake_embodiment")
        self.language_key = "annotation.language.language_instruction"
        self.modality_configs = {
            "action": SimpleNamespace(
                delta_indices=[0, 1],
                modality_keys=["arm", "gripper"],
            )
        }
        self.processor = SimpleNamespace(
            state_action_processor=SimpleNamespace(
                norm_params={
                    "fake_embodiment": {
                        "action": {
                            "arm": {"dim": np.array(2)},
                            "gripper": {"dim": np.array(1)},
                        }
                    }
                }
            )
        )
        FakeGr00tPolicy.instances.append(self)

    def get_action(self, obs, options=None):
        self.seen_obs = obs
        return {
            "arm": np.array([[[1.0, 2.0]]], dtype=np.float64),
            "gripper": [[[3.0]]],
        }, {"latency_ms": 1.0}

    def reset(self):
        self.reset_calls += 1
        return {"reset": True}


@pytest.fixture(autouse=True)
def fake_gr00t_policy(monkeypatch):
    FakeGr00tPolicy.instances.clear()
    monkeypatch.setattr(pipeline_gr00t, "Gr00tPolicy", FakeGr00tPolicy)


def _pipeline():
    od_config = SimpleNamespace(
        model="nvidia/GR00T-N1.7-3B",
        model_config={
            "embodiment_tag": "LIBERO_PANDA",
            "strict": False,
        },
        custom_pipeline_args={},
    )
    return Gr00tN1d7Pipeline(od_config=od_config)


def test_pipeline_initializes_local_policy():
    pipeline = _pipeline()

    policy = FakeGr00tPolicy.instances[0]
    assert policy.kwargs["model_path"] == "nvidia/GR00T-N1.7-3B"
    assert policy.kwargs["embodiment_tag"] == "LIBERO_PANDA"
    assert policy.kwargs["strict"] is False
    assert pipeline.weights_sources == ()
    assert pipeline.load_weights(iter(())) == set()


def test_forward_returns_dict_actions_in_output():
    pipeline = _pipeline()
    req = OmniDiffusionRequest(
        prompt="pick",
        request_id="req",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={
                "robot_obs": {
                    "images": {"cam": np.zeros((1, 1, 8, 8, 3), dtype=np.uint8)},
                    "state": {"joint": np.zeros((1, 1, 2), dtype=np.float32)},
                    "prompt": "pick the cube",
                    "session_id": "session-a",
                },
                "reset": True,
            }
        ),
    )

    output = pipeline.forward(DiffusionRequestBatch([req]))[0]

    assert output.error is None
    actions = output.output["actions"]
    assert set(actions) == {"arm", "gripper"}
    assert actions["arm"].dtype == np.float32
    np.testing.assert_allclose(actions["arm"], np.array([[[1.0, 2.0]]], dtype=np.float32))
    policy = FakeGr00tPolicy.instances[0]
    assert policy.seen_obs is not None
    assert "video" in policy.seen_obs
    assert policy.seen_obs["language"] == {"annotation.language.language_instruction": [["pick the cube"]]}
    assert "images" not in policy.seen_obs
    assert "prompt" not in policy.seen_obs
    assert "session_id" not in policy.seen_obs
    assert policy.reset_calls == 1


def test_dummy_warmup_returns_shape_correct_zero_actions():
    pipeline = _pipeline()
    req = OmniDiffusionRequest(
        prompt="dummy run",
        request_id="dummy_req_id",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )

    output = pipeline.forward(DiffusionRequestBatch([req]))[0]

    assert output.error is None
    actions = output.output["actions"]
    assert set(actions) == {"arm", "gripper"}
    assert actions["arm"].shape == (1, 2, 2)
    assert actions["gripper"].shape == (1, 2, 1)
    assert not actions["arm"].any()
    assert FakeGr00tPolicy.instances[0].seen_obs is None


def test_reset_delegates_to_policy():
    pipeline = _pipeline()

    assert pipeline.reset() == {"reset": True}
    assert FakeGr00tPolicy.instances[0].reset_calls == 1


class _BatchRecordingPolicy(FakeGr00tPolicy):
    def __init__(self):
        super().__init__()
        self.calls = []

    def get_action(self, obs, options=None):
        self.calls.append((obs, options))
        count = obs["video"]["cam"].shape[0]
        generators = options["generators"] if "generators" in options else [options["generator"]]
        noise = np.array([torch.randn((), generator=item).item() for item in generators], dtype=np.float32)
        pixels = obs["video"]["cam"][:, 0, 0, 0, 0].astype(np.float32)
        states = obs["state"]["joint"][:, 0, 0]
        prompts = np.array([len(row[0]) for row in obs["language"][self.language_key]], dtype=np.float32)
        value = (pixels + states + prompts + noise).reshape(count, 1, 1)
        return {
            "arm": np.broadcast_to(value, (count, 2, 2)).copy(),
            "gripper": np.broadcast_to(value, (count, 2, 1)).copy(),
        }, {}


def _batch_gr00t_pipeline():
    pipeline = _pipeline()
    pipeline.policy = _BatchRecordingPolicy()
    return pipeline


def _batch_gr00t_request(index, *, valid=True):
    observation = {
        "images": {"cam": np.full((1, 1, 8, 8, 3), 10 + index, dtype=np.uint8)},
        "state": {"joint": np.full((1, 1, 2), index + 1, dtype=np.float32)},
        "prompt": f"pick object {index}",
    }
    if not valid:
        observation["images"]["cam"] = np.zeros((2, 1, 8, 8, 3), dtype=np.uint8)
    return OmniDiffusionRequest(
        prompt=observation["prompt"],
        request_id=f"gr00t-{index}",
        sampling_params=OmniDiffusionSamplingParams(seed=300 + index, extra_args={"robot_obs": observation}),
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_gr00t_batch_fuses_requests_and_keeps_action_rows(batch_size):
    requests = [_batch_gr00t_request(i) for i in range(batch_size)]
    pipeline = _batch_gr00t_pipeline()

    results = pipeline.forward(DiffusionRequestBatch(requests))

    assert len(pipeline.policy.calls) == 1
    obs, options = pipeline.policy.calls[0]
    assert obs["video"]["cam"].shape[0] == obs["state"]["joint"].shape[0] == batch_size
    assert len(obs["language"][pipeline.policy.language_key]) == batch_size
    if batch_size == 1:
        assert isinstance(options["generator"], torch.Generator)
    else:
        assert len(options["generators"]) == batch_size
    assert len(results) == batch_size
    for index, request in enumerate(requests):
        assert results[index].error is None
        action = results[index].output["actions"]
        assert action["arm"].shape == (1, 2, 2) and action["arm"].dtype == np.float32
        assert action["gripper"].shape == (1, 2, 1)
        solo = _batch_gr00t_pipeline().forward(DiffusionRequestBatch([request]))[0]
        np.testing.assert_array_equal(action["arm"], solo.output["actions"]["arm"])

    reversed_results = _batch_gr00t_pipeline().forward(DiffusionRequestBatch(list(reversed(requests))))
    for original, reversed_output in zip(reversed(results), reversed_results):
        np.testing.assert_array_equal(original.output["actions"]["arm"], reversed_output.output["actions"]["arm"])


def test_gr00t_invalid_peer_keeps_valid_output():
    pipeline = _batch_gr00t_pipeline()
    good = _batch_gr00t_request(1)
    results = pipeline.forward(DiffusionRequestBatch([_batch_gr00t_request(0, valid=False), good]))

    assert results[0].error is not None
    assert results[1].error is None
    assert pipeline.policy.calls[0][0]["video"]["cam"].shape[0] == 1
    solo = _batch_gr00t_pipeline().forward(DiffusionRequestBatch([good]))[0]
    np.testing.assert_array_equal(results[1].output["actions"]["arm"], solo.output["actions"]["arm"])
