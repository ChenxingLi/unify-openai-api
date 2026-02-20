from functools import partial
import os

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from ..request_modifers.anthropic import OpenAIToAnthropicMiddleware
from ..request_modifers.open_webui import OpenWebUIRequest
from ..request_modifers.interface import SetModel
from ..response_handlers.cost_record import ChatCompletionCostRecord
from ..response_handlers.anthropic import AnthropicToOpenAI

from ..backends.openai import OpenAIProxy
from ..backends.anthropic import AnthropicProxy


from ..types.llm_api import LLMApi

        
def regiester_models(models: dict[str, LLMApi]):
    open_ai = AsyncOpenAI(api_key=os.getenv("DEERAPI_KEY"), base_url="https://api.deerapi.com/v1")
    anthropic = AsyncAnthropic(api_key=os.getenv("DEERAPI_KEY"), base_url="https://api.deerapi.com/")

    def openai_model(target, input_price, output_price):
        request_modifiers = [
            SetModel(model_name=target),
            OpenWebUIRequest()
        ]
        response_handlers = [
            ChatCompletionCostRecord(model_id=target, input_price=input_price, output_price=output_price),
        ]
        return OpenAIProxy(client=open_ai, request_modifiers=request_modifiers, response_handlers=response_handlers)


    def anthropic_model(target, input_price, output_price):
        request_modifiers = [
            SetModel(model_name=target),
            OpenWebUIRequest(chat_completion_request = False),
            OpenAIToAnthropicMiddleware()
        ]
        response_handlers = [
            AnthropicToOpenAI(),
            ChatCompletionCostRecord(model_id=target, input_price=input_price, output_price=output_price)
        ]
        return AnthropicProxy(client=anthropic, request_modifiers=request_modifiers, response_handlers=response_handlers)

    # ===== OpenAI (GPT) =====
    # GPT-5.2 系列
    models["gpt-5.2"] = openai_model("gpt-5.2", input_price=1.75, output_price=14)
    models["gpt-5.2-codex"] = openai_model("gpt-5.2-codex", input_price=1.75, output_price=14)
    models["gpt-5.2-chat"] = openai_model("gpt-5.2-chat-latest", input_price=1.75, output_price=14)
    models["gpt-5.2-pro"] = openai_model("gpt-5.2-pro", input_price=21, output_price=168)
    # GPT-4.5
    models["gpt-4.5"] = openai_model("gpt-4.5-preview-2025-02-27", input_price=375, output_price=750)
    # GPT-4o 系列
    models["gpt-4o"] = openai_model("gpt-4o-2024-11-20", input_price=12.5, output_price=50)

    # ===== Anthropic (Claude) =====
    # Claude Opus 4.6
    models["claude-opus-4.6"] = anthropic_model("claude-opus-4-6", input_price=5, output_price=25)
    # Claude Opus 4.5
    models["claude-opus-4.5"] = anthropic_model("claude-opus-4-5-20251101", input_price=5, output_price=25)
    models["claude-opus-4.5-think"] = anthropic_model("claude-opus-4-5-20251101-thinking", input_price=5, output_price=25)
    # Claude Sonnet 4.5
    models["claude-sonnet-4.5"] = anthropic_model("claude-sonnet-4-5-20250929", input_price=3, output_price=15)
    models["claude-sonnet-4.5-think"] = anthropic_model("claude-sonnet-4-5-20250929-thinking", input_price=3, output_price=15)
    # Claude Haiku 4.5
    models["claude-haiku-4.5"] = anthropic_model("claude-haiku-4-5-20251001", input_price=1, output_price=5)

    # ===== Google (Gemini) =====
    # Gemini 3 系列
    models["gemini-3.1-pro"] = openai_model("gemini-3.1-pro-preview", input_price=2, output_price=12)
    models["gemini-3.1-pro-think"] = openai_model("gemini-3.1-pro-preview-thinking", input_price=2, output_price=12)
    models["gemini-3-pro"] = openai_model("gemini-3-pro-preview", input_price=2, output_price=12)
    models["gemini-3-pro-think"] = openai_model("gemini-3-pro-preview-thinking", input_price=2, output_price=12)
    models["gemini-3-flash"] = openai_model("gemini-3-flash-preview", input_price=0.5, output_price=3)

    # ===== xAI (Grok) =====
    # Grok-4 系列
    models["grok-4"] = openai_model("grok-4", input_price=3, output_price=15)
    models["grok-4-fast"] = openai_model("grok-4-fast-reasoning", input_price=0.2, output_price=0.5)
        
        