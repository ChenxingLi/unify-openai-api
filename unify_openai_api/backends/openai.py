from dataclasses import dataclass

from openai import AsyncOpenAI

from ..utils.split_params import split_params
from .base_chat_completion import BaseChatCompletion

@dataclass
class OpenAIProxy(BaseChatCompletion):
    client: AsyncOpenAI
    
    def _make_request_inner(self, data):
        support_fields, extra_fields = split_params(OPENAI_FIELD_SET, data)
        support_fields.setdefault("extra_body", dict()).update(extra_fields)
        return self.client.chat.completions.create(**support_fields)

OPENAI_FIELD_SET = {
    "messages",
    "model",
    "audio",
    "frequency_penalty",
    "function_call",
    "functions",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "metadata",
    "modalities",
    "n",
    "parallel_tool_calls",
    "prediction",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "seed",
    "service_tier",
    "stop",
    "store",
    "stream",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "user",
    "extra_headers",
    "extra_query",
    "extra_body",
    "timeout"
}