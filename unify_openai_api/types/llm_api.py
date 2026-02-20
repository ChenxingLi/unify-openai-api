from typing import abstractmethod
from abc import ABC

from unify_openai_api.types.response import ApiResponse
from typing import Dict, TypedDict, cast
from fastapi import Request

from ..usage_db.writer import AsyncDBWriter


class LLMApi(ABC):
    @abstractmethod
    async def make_request(self, data: dict) -> ApiResponse: ...
    
    @abstractmethod
    async def handle_response(self, obj: ApiResponse, state: 'AppState'): ...


class AppState(TypedDict, total=False):
    models: Dict[str, LLMApi]
    writer: AsyncDBWriter


def get_typed_state(request: Request) -> AppState:
    # return AppState(models=request.state.models, writer=request.state.writer)  # type: ignore
    return cast(AppState, request.app.state)