from dataclasses import dataclass

from .interface import ResponseHandler

from ..types.state import AppState
from ..usage_db.writer import AsyncDBWriter

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.completion import CompletionUsage


@dataclass
class ChatCompletionCostRecord(ResponseHandler):
    model_id: str
    input_price: float
    output_price: float

    def handle_response(self, state: AppState, user_id: str, data: ChatCompletion) -> ChatCompletion:
        if data.usage:
            self._add_usage(state.writer, user_id, data.usage)

        return data
    
    def handle_response_frame(self, state: AppState, user_id: str, frame: ChatCompletionChunk) -> ChatCompletionChunk:        
        if frame.usage:
            self._add_usage(state.writer, user_id, frame.usage)
        
        return frame
    
    def _add_usage(self, writer: AsyncDBWriter, user_id: str, usage: CompletionUsage):
        writer.add_usage(model_id = self.model_id, 
            input_tokens = usage.prompt_tokens,
            output_tokens = usage.completion_tokens,
            input_price = self.input_price,
            output_price = self.output_price,
            user_id = user_id,
        )