from typing import Any, Dict, TypedDict, cast
from fastapi import Request, Depends

from .llm_api import LLMApi
from ..usage_db.writer import AsyncDBWriter

class AppState(TypedDict, total=False):
    models: Dict[str, LLMApi]
    writer: AsyncDBWriter


def get_typed_state(request: Request) -> AppState:
    return cast(AppState, request.state)
