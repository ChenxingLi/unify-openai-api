from anthropic.types.beta import BetaRawMessageStopEvent
from anthropic import AsyncAnthropic
from anthropic.lib.streaming import ParsedBetaMessageStreamEvent
from anthropic.types import Message, MessageParam, Usage, ThinkingConfigEnabledParam, ThinkingConfigDisabledParam
from anthropic._exceptions import AnthropicError
import json
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from dataclasses import dataclass
from typing import Any, Callable, Awaitable, Optional, List, Dict

from openai.types.completion_usage import CompletionUsage


import logging

from . import ChatCompletion
from ..usage_db.writer import AsyncDBWriter

# 每个模块获取自己的logger（自动继承根配置）
logger = logging.getLogger(__name__)


@dataclass
class BaseAnthropicResponse:
    response: Any
    user_id: Optional[str]
    stream: bool


@dataclass
class BaseAnthropicModel(ChatCompletion):
    client: AsyncAnthropic

    async def make_request(self, data: dict):
        stream = data.pop("stream", False)

        openwebui_meta = data.pop("openwebui_middleware", None)

        if openwebui_meta is not None:
            user_id = openwebui_meta.get("user_id", None)
        else:
            user_id = None

        # 将 messages 转换为 Anthropic 的 MessageParam，如果需要
        data["messages"] = [MessageParam(**msg) for msg in data.pop("messages", [])]

        # Anthropic 使用 'max_tokens' 而非 'max_completion_tokens'，但支持 'max_tokens'
        if "max_completion_tokens" in data:
            data["max_tokens"] = data.pop("max_completion_tokens")
        
        data.setdefault("max_tokens", 32768)
            
        # 添加 budger tokens
        if "budget_tokens" in data:
            budget_tokens = data.pop("budget_tokens")
            if budget_tokens > 0:
                data["thinking"] = ThinkingConfigEnabledParam(budget_tokens=budget_tokens, type = "enabled")
            else:
                data["thinking"] = ThinkingConfigDisabledParam(type = "disabled")
                
        if "reasoning_effort" in data:
            # TODO
            pass
            

        # 分离参数：Anthropic 参数与 OpenAI 类似但不完全相同，需要调整
        # 例如，Anthropic 使用 'messages' (List[MessageParam])，'model'，'max_tokens' 等
        # 忽略不支持的参数，如 'n' (Anthropic 只支持 n=1)，'logprobs' 等
        support_fields, extra_fields = split_anthropic_params(data)  # 假设 split_openai_params 已适配或需修改以匹配 Anthropic

        # extra_body 可以用于额外参数
        support_fields.setdefault("extra_body", dict()).update(extra_fields)

        # 调用 Anthropic 的 messages.create
        response = await self.client.messages.create(**support_fields, stream=stream)

        return BaseAnthropicResponse(response=response, stream=stream, user_id=user_id)

    async def handle_response(self, obj: BaseAnthropicResponse, state):
        response = obj.response  # 对于非流式，已是 awaitable 的结果；对于流式，是 AsyncMessageStream
        writer: AsyncDBWriter = state.writer

        if not obj.stream:
            try:
                logger.debug(f"response: {response}")
                if hasattr(response, 'usage') and response.usage is not None:
                    # 映射到 OpenAI 的 CompletionUsage 格式
                    usage = CompletionUsage(
                        prompt_tokens=response.usage.input_tokens,
                        completion_tokens=response.usage.output_tokens,
                        total_tokens=response.usage.input_tokens + response.usage.output_tokens
                    )
                    self.record_token_usage(obj.user_id, usage, writer)
                # 转换为 OpenAI 兼容的响应格式
                return to_openai_format(response)
            except AnthropicError as e:
                logger.warning(f"Anthropic API error: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Anthropic API error: {e}")
        else:
            handle_usage = lambda usage: self.record_token_usage(obj.user_id, usage, writer)
            return StreamingResponse(response_stream(response, handle_usage), media_type="text/event-stream")

    def record_token_usage(self, user_id: Optional[str], usage: CompletionUsage, writer: AsyncDBWriter):
        pass


@dataclass
class AnthropicProxy(BaseAnthropicModel):
    name: str
    target: str
    input_price: float
    output_price: float

    @property
    def model_id(self):
        return f"{self.name}"

    async def make_request(self, data: dict):
        data["model"] = self.target
        return await super().make_request(data)

    def record_token_usage(self, user_id: Optional[str], usage: CompletionUsage, writer: AsyncDBWriter):
        writer.add_usage(model_id=self.target,
                         input_tokens=usage.prompt_tokens,
                         output_tokens=usage.completion_tokens,
                         input_price=self.input_price,
                         output_price=self.output_price,
                         user_id=user_id,
                         )
        # logger.info(f"usage: Input {usage.prompt_tokens} Output {usage.completion_tokens}")


async def response_stream(response, handle_usage: Callable[[Usage], None]):
    try:
        async for event in response:
            logger.debug(f"event: {event}")
            # Anthropic 的流事件类型：MessageStartEvent, ContentBlockStartEvent, ContentBlockDeltaEvent, MessageDeltaEvent, MessageStopEvent 等
            # 构建 OpenAI 兼容的 chunk 格式
            chunk = to_openai_chunk_format(event)
            if chunk and chunk.usage:
                handle_usage(chunk.usage)
            if chunk:
                yield f"data: {json.dumps(chunk.model_dump())}\n\n"
        # 发送结束标记
        yield "data: [DONE]\n\n"
    except AnthropicError as e:
        logger.warning(f"Stream error: {e}")
        yield f"data: {json.dumps({'error': f'stream error: {e}'})}\n\n"
    except Exception as e:
        logger.warning(f"Unexpected error during streaming: {e}")
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'error': f'Internal server error: {e}'})}\n\n"


def to_openai_format(anthropic_response: Message) -> Dict:
    # 将 Anthropic Message 转换为 OpenAI ChatCompletion 格式
    return {
        "id": anthropic_response.id,
        "choices": [
            {
                "finish_reason": CLAUDE_TO_OPENAI_FINISH.get(anthropic_response.stop_reason, None),
                "index": 0,
                "message": {
                    "content": "".join(block.text for block in anthropic_response.content if block.type == "text"),
                    "role": "assistant"
                }
            }
        ],
        # Anthropic 无 created
        "created": int(anthropic_response.created_at.timestamp()) if hasattr(anthropic_response, 'created_at') else None,
        "model": anthropic_response.model,
        "object": "chat.completion",
        "usage": {
            "prompt_tokens": anthropic_response.usage.input_tokens,
            "completion_tokens": anthropic_response.usage.output_tokens,
            "total_tokens": anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens
        } if anthropic_response.usage else None
    }

import time
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from openai.types.completion_usage import CompletionUsage

CLAUDE_TO_OPENAI_FINISH = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "content_filter",
    None: None  # 处理 null 值
}

def to_openai_chunk_format(event: ParsedBetaMessageStreamEvent, message_id="", model="") -> Optional[ChatCompletionChunk]:
    """
    根据 Anthropic 事件类型转换为 OpenAI ChatCompletionChunk 对象
    """
    if event.type == "message_start":
        return ChatCompletionChunk(
            id=event.message.id,
            choices=[Choice(
                delta=ChoiceDelta(role="assistant"),
                finish_reason=None,
                index=0,
                logprobs=None
            )],
            created=int(time.time()),
            model=event.message.model,
            object="chat.completion.chunk",
            service_tier=None,
            system_fingerprint=None,
            usage=None
        )
    
    elif event.type == "content_block_delta":
        if event.delta.type == "thinking_delta":
            return ChatCompletionChunk(
                id=message_id,
                choices=[Choice(
                    delta=ChoiceDelta(
                        content=None,
                        function_call=None,
                        refusal=None,
                        role=None,
                        tool_calls=None,
                        reasoning_content=event.delta.thinking
                    ),
                    finish_reason=None,
                    index=0,
                    logprobs=None
                )],
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk",
                service_tier=None,
                system_fingerprint=None,
                usage=None
            )
        elif event.delta.type == "text_delta":
            return ChatCompletionChunk(
                id=message_id,
                choices=[Choice(
                    delta=ChoiceDelta(
                        content=event.delta.text,
                        function_call=None,
                        refusal=None,
                        role=None,
                        tool_calls=None,
                        reasoning_content=None
                    ),
                    finish_reason=None,
                    index=0,
                    logprobs=None
                )],
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk",
                service_tier=None,
                system_fingerprint=None,
                usage=None
            )
        return None
    
    elif event.type == "message_delta":
        claude_stop_reason = event.delta.stop_reason
        openai_finish_reason = CLAUDE_TO_OPENAI_FINISH.get(claude_stop_reason, None)
        
        # 提取 token usage（仅在最终 delta 中存在）
        usage = None
        if hasattr(event, 'usage') and event.usage:
            prompt_tokens = (
                event.usage.input_tokens 
                + (event.usage.cache_read_input_tokens or 0) 
                + (event.usage.cache_creation_input_tokens or 0)
            )
            completion_tokens = event.usage.output_tokens
            total_tokens = prompt_tokens + completion_tokens
            usage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        return ChatCompletionChunk(
            id=message_id,
            choices=[Choice(
                delta=ChoiceDelta(),
                finish_reason=openai_finish_reason,
                index=0,
                logprobs=None
            )],
            created=int(time.time()),
            model=model,
            object="chat.completion.chunk",
            service_tier=None,
            system_fingerprint=None,
            usage=usage,
        )
    
    return None

def split_anthropic_params(data) -> tuple[dict, dict]:
    supported_fields = {
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
    supported, unsupported = {}, {}
    for key, value in data.items():
        if key in supported_fields:
            supported[key] = value
        else:
            unsupported[key] = value
    return supported, unsupported
