from openai import AsyncOpenAI
from openai._exceptions import OpenAIError
from openai.types.completion_usage import CompletionUsage
import json
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from dataclasses import dataclass
from typing import Any, Callable, Awaitable, Optional

import logging

from . import ChatCompletion
from ..openai_utils import split_openai_params
from ..usage_db.writer import AsyncDBWriter

# 每个模块获取自己的logger（自动继承根配置）
logger = logging.getLogger(__name__)


@dataclass
class BaseOpenAIResponse:
    response: Any
    user_id: Optional[str]
    stream: bool

@dataclass
class BaseOpenAIModel(ChatCompletion):
    client: AsyncOpenAI
    
    async def make_request(self, data: dict):
        stream = data.get("stream", False)
        
        openwebui_meta = data.pop("openwebui_middleware", None)
        
        if openwebui_meta is not None:
            user_id = openwebui_meta.get("user_id", None)
        else:
            user_id = None
        
        if stream:
            data.setdefault("stream_options", dict())["include_usage"] = True
            
        support_fields, extra_fields = split_openai_params(data)
        support_fields.setdefault("extra_body", dict()).update(extra_fields)
        response = self.client.chat.completions.create(**support_fields)
        
        return BaseOpenAIResponse(response = response, stream = stream, user_id = user_id)
    
    async def handle_response(self, obj: BaseOpenAIResponse, state):
        response = await obj.response
        writer: AsyncDBWriter = state.writer
        
        if not obj.stream:
            try:
                logger.debug(f"response: {response}")
                if response.usage is not None:
                    self.record_token_usage(obj.user_id, response.usage, writer)
                return response.model_dump()
            except OpenAIError as e:
                logger.warning(f"OpenAI API error: {e}")
                raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")
        else:
            handle_usage = lambda usage: self.record_token_usage(obj.user_id, usage, writer)
            return StreamingResponse(response_stream(response, handle_usage), media_type="text/event-stream")
        
    def record_token_usage(self, user_id: Optional[str], usage: CompletionUsage, writer: AsyncDBWriter):
        pass

@dataclass
class OpenAIProxy(BaseOpenAIModel):
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
        writer.add_usage(model_id = self.target, 
                         input_tokens = usage.prompt_tokens,
                         output_tokens = usage.completion_tokens,
                         input_price = self.input_price,
                         output_price = self.output_price,
                         user_id = user_id,
                         )
        # logger.info(f"usage: Input {usage.prompt_tokens} Output {usage.completion_tokens}")
    

async def response_stream(response, handle_usage: Callable[[CompletionUsage], None]):
    try:
        async for chunk in response:
            logger.debug(f"chunk: {chunk}")
            if chunk.usage is not None:
                handle_usage(chunk.usage)
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"
        # 可选：发送结束标记
        # yield "data: [DONE]\n\n"
    except OpenAIError as e:
        logger.warning(f"Stream error: {e}")
        # 可选：发送错误信息到流中
        yield f"data: {json.dumps({'error': f'stream error: {e}'})}\n\n"
    except Exception as e:
        logger.warning(f"Unexpected error during streaming: {e}")
        yield f"data: {json.dumps({'error': f'Internal server error: {e}'})}\n\n"