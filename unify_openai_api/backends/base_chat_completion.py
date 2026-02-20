from abc import abstractmethod

from openai._exceptions import OpenAIError
from anthropic._exceptions import AnthropicError

from openai.types.chat import ChatCompletion
import json
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from dataclasses import dataclass, field
from typing import Any, List

import logging

from unify_openai_api.types.state import AppState

from ..request_modifers.interface import RequestModifier, modify_request
from ..response_handlers.interface import ResponseHandler, handle_response, handle_response_frames
from ..types.response import ApiResponse
from ..types.llm_api import LLMApi

import traceback

logger = logging.getLogger(__name__)


@dataclass
class BaseChatCompletion(LLMApi):
    request_modifiers: List[RequestModifier]
    response_handlers: List[ResponseHandler]
    
    @abstractmethod
    def _make_request_inner(self, data: dict) -> Any: ...

    async def make_request(self, data: dict) -> ApiResponse:
        data = modify_request(self.request_modifiers, data)

        stream = data.get("stream", False)
        user_id = data.get("user_id", None)

        response = self._make_request_inner(data)
        
        return ApiResponse(response = response, stream = stream, user_id = user_id)
    
    async def handle_response(self, obj: ApiResponse, state: AppState):
        response: ChatCompletion = await obj.response
        
        if not obj.stream:
            try:
                logger.debug(f"response: {response}")
                handle_response(self.response_handlers, state, obj.user_id, response)
                return response.model_dump()
            except OpenAIError as e:
                logger.warning(f"OpenAI API error: {e}")
                raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")
        else:  
            return StreamingResponse(self._response_stream(obj, state, response), media_type="text/event-stream")

    async def _response_stream(self, obj: ApiResponse, state: AppState, response: ChatCompletion):
        try:
            async for event in response:
                logger.debug(f"event: {event}")
                event = handle_response_frames(self.response_handlers, state, obj.user_id, event)
                if event is not None:
                    yield f"data: {json.dumps(event.model_dump())}\n\n"
            # 可选：发送结束标记
            # yield "data: [DONE]\n\n"
        except OpenAIError as e:
            logger.warning(f"Stream error: {e}")
            # 可选：发送错误信息到流中
            yield f"data: {json.dumps({'error': f'stream error: {e}'})}\n\n"
        except AnthropicError as e:
            logger.warning(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': f'stream error: {e}'})}\n\n"
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"Unexpected error during streaming: {e}")
            yield f"data: {json.dumps({'error': f'Internal server error: {e}'})}\n\n"    