from anthropic.lib.streaming import ParsedBetaMessageStreamEvent
from anthropic.types import Message
from dataclasses import dataclass
from typing import Optional, Dict

import time
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from openai.types.completion_usage import CompletionUsage

from ..response_handlers.interface import ResponseHandler

@dataclass
class AnthropicToOpenAI(ResponseHandler):
    def handle_response(self, state, user_id: str, data: Message) -> Dict:
        return to_openai_format(data)
    
    def handle_response_frame(self, state, user_id: str, frame: ParsedBetaMessageStreamEvent) -> Optional[ChatCompletionChunk]:
        return to_openai_chunk_format(frame)


def to_openai_format(anthropic_response: Message) -> Dict:
    # 将 Anthropic Message 转换为 OpenAI LLMApi 格式
    # Not tested
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