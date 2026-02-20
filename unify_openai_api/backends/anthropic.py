from anthropic import AsyncAnthropic
from dataclasses import dataclass
from typing import Any

from .base_chat_completion import BaseChatCompletion
from ..utils.split_params import split_params


@dataclass
class AnthropicProxy(BaseChatCompletion):
    client: AsyncAnthropic

    def _make_request_inner(self, data: dict) -> Any:
        stream = data.pop("stream", False)

        # 假设 split_openai_params 已适配或需修改以匹配 Anthropic
        support_fields, extra_fields = split_params(ANTHROPIC_FIELD_SET, data)

        # extra_body 可以用于额外参数
        support_fields.setdefault("extra_body", dict()).update(extra_fields)


        # 调用 Anthropic 的 messages.create
        return self.client.messages.create(**support_fields, stream=stream)


ANTHROPIC_FIELD_SET = {
    "max_tokens",
    "messages",
    "model",
    "stream",
    "container",
    "inference_geo",
    "metadata",
    "output_config",
    "service_tier",
    "stop_sequences",
    "system",
    "temperature",
    "thinking",
    "tool_choice",
    "tools",
    "top_k",
    "top_p",
    "extra_headers",
    "extra_query",
    "extra_body",
    "timeout"
}
