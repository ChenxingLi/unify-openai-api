import os
from openai import AsyncOpenAI
from dataclasses import dataclass
from .openai import OpenAIProxy
from . import ChatCompletion
        
@dataclass
class DoubaoModel(OpenAIProxy):
    pass
    
        
def regiester_models(models: dict[str, ChatCompletion]):
    client = AsyncOpenAI(api_key=os.getenv("VOLC_API_KEY"), base_url="https://ark.cn-beijing.volces.com/api/v3")
    
    def add_model(name, target, input_price, output_price):
        model = DoubaoModel(client=client, name=name, target=target, input_price=input_price, output_price=output_price)
        models[model.model_id] = model
    
    add_model("Doubao-pro-32k", "doubao-1-5-pro-32k-250115", 0.8, 2)
    add_model("Doubao-thinking-pro", "doubao-1-5-thinking-pro-250415", 4, 16)
    add_model("deepseek-r1", "deepseek-r1-250120", 4, 16)
    add_model("deepseek-v3", "deepseek-v3-250324", 2, 8)
    add_model("Doubao-vision-pro-32k", "doubao-1-5-vision-pro-32k-250115", 3, 9)
    add_model("Doubao-pro-256k", "doubao-1-5-pro-256k-250115", 5, 9)
    
    