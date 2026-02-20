import os
from openai import AsyncOpenAI
from dataclasses import dataclass
from .openai import OpenAIProxy
from .anthropic import AnthropicProxy
from . import ChatCompletion
from anthropic import AsyncAnthropic

        
@dataclass
class DeerApiModel(OpenAIProxy):
    pass
    
@dataclass
class DeerApiAnthropicModel(AnthropicProxy):
    pass
        
def regiester_models(models: dict[str, ChatCompletion]):
    client = AsyncOpenAI(api_key=os.getenv("DEERAPI_KEY"), base_url="https://api.deerapi.com/v1")
    a_client = AsyncAnthropic(api_key=os.getenv("DEERAPI_KEY"), base_url="https://api.deerapi.com/")
    
    def add_model(name, target, input_price, output_price):
        model = DeerApiModel(client=client, name=name, target=target, input_price=input_price, output_price=output_price)
        models[model.model_id] = model
        
    def add_anthropic_model(name, target, input_price, output_price):
        model = DeerApiAnthropicModel(client=a_client, name=name, target=target, input_price=input_price, output_price=output_price)
        models[model.model_id] = model

    # ===== OpenAI (GPT) =====
    # GPT-5.2 系列
    add_model("gpt-5.2", "gpt-5.2", 1.75, 14)
    add_model("gpt-5.2-codex", "gpt-5.2-codex", 1.75, 14)
    add_model("gpt-5.2-chat", "gpt-5.2-chat-latest", 1.75, 14)
    add_model("gpt-5.2-pro", "gpt-5.2-pro", 21, 168)

    # GPT-4.5
    add_model("gpt-4.5", "gpt-4.5-preview-2025-02-27", 375, 750)

    # GPT-4o 系列
    add_model("gpt-4o", "gpt-4o-2024-11-20", 12.5, 50)
    # add_model("gpt-4o-all", "gpt-4o-all", 12.5, 50)

    # ===== Anthropic (Claude) =====

    # Claude Opus 4.6
    add_anthropic_model("claude-opus-4.6", "claude-opus-4-6", 5, 25)

    # Claude Opus 4.5
    add_anthropic_model("claude-opus-4.5", "claude-opus-4-5-20251101", 5, 25)
    add_anthropic_model("claude-opus-4.5-think", "claude-opus-4-5-20251101-thinking", 5, 25)

    # Claude Sonnet 4.5
    add_anthropic_model("claude-sonnet-4.5", "claude-sonnet-4-5-20250929", 3, 15)
    add_anthropic_model("claude-sonnet-4.5-think", "claude-sonnet-4-5-20250929-thinking", 3, 15)

    # Claude Haiku 4.5
    add_anthropic_model("claude-haiku-4.5", "claude-haiku-4-5-20251001", 1, 5)
    
    
    
    # Claude 3.7
    # add_anthropic_model("claude-3.7", "claude-3-7-sonnet-20250219", 15, 75)
    # add_anthropic_model("claude-3.7-think", "claude-3-7-sonnet-thinking", 30, 150)


    # ===== Google (Gemini) =====
    # Gemini 3 系列
    add_model("gemini-3.1-pro", "gemini-3.1-pro-preview", 2, 12)
    add_model("gemini-3.1-pro-think", "gemini-3.1-pro-preview-thinking", 2, 12)
    add_model("gemini-3-pro", "gemini-3-pro-preview", 2, 12)
    add_model("gemini-3-pro-think", "gemini-3-pro-preview-thinking", 2, 12)
    add_model("gemini-3-flash", "gemini-3-flash-preview", 0.5, 3)

    # # Gemini 2.5 系列
    # add_model("gemini-2.5-pro", "gemini-2.5-pro-preview-06-05", 6.25, 50)
    # add_model("gemini-2.5-flash", "gemini-2.5-flash-preview-05-20", 0.75, 17.5)


    # ===== xAI (Grok) =====
    # Grok-4 系列
    add_model("grok-4", "grok-4", 3, 15)
    add_model("grok-4-fast", "grok-4-fast-reasoning", 0.2, 0.5)
    # add_model("grok-4-fast-no-think", "grok-4-fast-non-reasoning", 0.2, 0.5)

    # Grok-3 系列
    # add_model("grok-3", "grok-3", 15, 75)
    # add_model("grok-3-reasoner", "grok-3-reasoner", 10, 40)
    # add_model("grok-3-deepsearch", "grok-3-deepsearch", 10, 40)
    # add_model("grok-3-fast", "grok-3-fast", 25, 125)
        
        