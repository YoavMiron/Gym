# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gateway reasoning-item normaliser: LiteLLM-style Responses bodies validate as NeMoGymResponse."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_models.openai_model.app import (
    SimpleModelServer,
    SimpleModelServerConfig,
    normalize_reasoning_items,
)


FIXTURES = Path(__file__).parent / "fixtures"
# Response bodies captured on 2026-10-16 from a LiteLLM gateway (Responses API, reasoning.effort=high),
# with ids, encrypted content, thinking signatures and the long texts shortened; structure untouched.
GATEWAY_BODIES = {
    "claude-opus-5-5": FIXTURES / "litellm_reasoning_claude_opus_5_5.json",
    "gemini-4-argon": FIXTURES / "litellm_reasoning_gemini_4_argon.json",
}


def _load(name: str) -> dict:
    return json.loads(GATEWAY_BODIES[name].read_text(encoding="utf-8"))


def _server(**kwargs) -> SimpleModelServer:
    config = SimpleModelServerConfig(
        host="0.0.0.0",
        port=8081,
        openai_base_url="https://gateway.invalid/v1",
        openai_api_key="dummy_key",  # pragma: allowlist secret
        openai_model="dummy_model",
        entrypoint="",
        name="test_model_server",
        **kwargs,
    )
    return SimpleModelServer(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))


@pytest.mark.parametrize("name", sorted(GATEWAY_BODIES))
class TestGatewayBodies:
    def test_raw_body_fails_with_exactly_the_two_documented_errors(self, name: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            NeMoGymResponse.model_validate(_load(name))

        assert sorted((error["loc"], error["type"]) for error in excinfo.value.errors()) == [
            (("output", 0, "reasoning", "content", 0, "type"), "literal_error"),
            (("output", 0, "reasoning", "summary"), "list_type"),
        ]

    async def test_server_repairs_the_body_by_default_and_keeps_usage(self, name: str) -> None:
        body = _load(name)
        pristine = deepcopy(body)
        server = _server()
        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_response = AsyncMock(return_value=body)

        response = await server.responses(NeMoGymResponseCreateParamsNonStreaming(input="hi"))

        reasoning, message = response.output
        assert (reasoning.type, reasoning.summary) == ("reasoning", [])
        assert [(part.type, part.text) for part in reasoning.content] == [
            ("reasoning_text", body["output"][0]["content"][0]["text"])
        ]
        assert reasoning.encrypted_content == body["output"][0]["encrypted_content"]
        # The answer message is not a reasoning item, so its output_text part keeps its type.
        assert (message.type, message.content[0].type) == ("message", "output_text")
        assert message.content[0].text == body["output"][1]["content"][0]["text"]
        usage = response.usage
        assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (
            body["usage"]["input_tokens"],
            body["usage"]["output_tokens"],
            body["usage"]["total_tokens"],
        )
        assert (
            usage.output_tokens_details.reasoning_tokens == body["usage"]["output_tokens_details"]["reasoning_tokens"]
        )
        assert body == pristine  # the upstream dict is never mutated

    async def test_opting_out_surfaces_the_raw_shape(self, name: str) -> None:
        server = _server(normalize_reasoning_items=False)
        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_response = AsyncMock(return_value=_load(name))

        with pytest.raises(ValidationError):
            await server.responses(NeMoGymResponseCreateParamsNonStreaming(input="hi"))


class TestNormalizeReasoningItems:
    def test_only_the_two_gateway_deviations_change(self) -> None:
        body = {
            "output": [
                {
                    "type": "reasoning",
                    "id": "r",
                    "summary": None,
                    "encrypted_content": "enc",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "t", "annotations": []},
                        {"type": "reasoning_text", "text": "u"},
                        "not-a-part",
                    ],
                },
                {
                    "type": "message",
                    "id": "m",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "answer", "annotations": []}],
                },
                "not-an-item",
            ],
            "usage": {"output_tokens": 3},
        }
        pristine = deepcopy(body)

        out = normalize_reasoning_items(body)

        assert body == pristine
        assert out["output"][0] == {
            "type": "reasoning",
            "id": "r",
            "summary": [],
            "encrypted_content": "enc",
            "status": "completed",
            "content": [
                {"type": "reasoning_text", "text": "t", "annotations": []},
                {"type": "reasoning_text", "text": "u"},
                "not-a-part",
            ],
        }
        assert out["output"][1:] == pristine["output"][1:]
        assert out["usage"] == {"output_tokens": 3}

    def test_missing_summary_becomes_an_empty_list(self) -> None:
        assert normalize_reasoning_items({"output": [{"type": "reasoning", "id": "r"}]}) == {
            "output": [{"type": "reasoning", "id": "r", "summary": []}]
        }

    @pytest.mark.parametrize(
        "body",
        [
            {
                "output": [
                    {
                        "type": "reasoning",
                        "id": "r",
                        "summary": [],
                        "content": [{"type": "reasoning_text", "text": "t"}],
                    }
                ]
            },
            {"output": [{"type": "message", "id": "m", "content": [{"type": "output_text", "text": "a"}]}]},
            {"output": "not-a-list"},
            {"id": "no-output"},
        ],
        ids=["well_formed_reasoning", "message_only", "non_list_output", "no_output"],
    )
    def test_everything_else_passes_through_unchanged(self, body: dict) -> None:
        assert normalize_reasoning_items(body) == deepcopy(body)
