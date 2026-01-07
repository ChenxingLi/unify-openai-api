import os
from openai import AsyncOpenAI
from dataclasses import dataclass
from .openai import OpenAIProxy
from . import ChatCompletion
        
@dataclass
class DeerApiModel(OpenAIProxy):
    pass
    
        
def regiester_models(models: dict[str, ChatCompletion]):
    client = AsyncOpenAI(api_key=os.getenv("DEERAPI_KEY"), base_url="https://api.deerapi.com/v1")
    
    def add_model(name, target, input_price, output_price):
        model = DeerApiModel(client=client, name=name, target=target, input_price=input_price, output_price=output_price)
        models[model.model_id] = model
        

    add_model("claude-3.7", "claude-3-7-sonnet-20250219", 15, 75)
    add_model("claude-3.7-think", "claude-3-7-sonnet-thinking", 30, 150)
    add_model("gemini-2.5-pro", "gemini-2.5-pro-preview-06-05", 6.25, 50)
    add_model("gemini-2.5-flash", "gemini-2.5-flash-preview-05-20", 0.75, 17.5)
    add_model("gpt-4.5", "gpt-4.5-preview-2025-02-27", 375, 750)
    add_model("gpt-4o", "gpt-4o-2024-11-20", 12.5, 50)
    add_model("gpt-4o-all", "gpt-4o-all", 12.5, 50)
    add_model("gpt-4-turbo", "gpt-4-turbo-2024-04-09", 50, 150)
    add_model("gpt-o1-pro", "o1-pro", 90, 360)
    add_model("gpt-o1", "o1-preview", 75, 300)
    add_model("gpt-4.1", "gpt-4.1", 10, 40)
    add_model("gpt-4.1-mini", "gpt-4.1-mini", 2, 8)
    add_model("gpt-4.1-nano", "gpt-4.1-nano", 0.5, 2.0)
    add_model("grok-3", "grok-3", 15, 75)
    add_model("grok-3-reasoner", "grok-3-reasoner", 10, 40)
    add_model("grok-3-deepsearch", "grok-3-deepsearch", 10, 40)
    add_model("grok-3-fast", "grok-3-fast", 25, 125)
    add_model("gpt-o3", "o3-2025-04-16", 10, 40)
    add_model("gpt-o3-pro", "o3-pro-2025-06-10", 100, 400)
    add_model("gpt-o4-mini", "o4-mini-2025-04-16", 5.5, 22)
    
    