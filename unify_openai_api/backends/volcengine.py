import os
from typing import Optional
from openai import AsyncOpenAI
from dataclasses import dataclass
from .openai import OpenAIProxy
from . import ChatCompletion
        
@dataclass
class DoubaoModel(OpenAIProxy):
    reasoning_effort: Optional[str]
    
    async def make_request(self, data: dict):
        if self.reasoning_effort:
            data["reasoning_effort"] = self.reasoning_effort
        return await super().make_request(data)
    pass
    
        
def regiester_models(models: dict[str, ChatCompletion]):
    client = AsyncOpenAI(api_key=os.getenv("VOLC_API_KEY"), base_url="https://ark.cn-beijing.volces.com/api/v3")
    
    def add_model(name, target, input_price, output_price, reasoning_effort = None):
        model = DoubaoModel(client=client, name=name, target=target, input_price=input_price / 7, output_price=output_price / 7, reasoning_effort=reasoning_effort)
        models[model.model_id] = model
    
    # TODO: 支持豆包动态定价长度
    # TODO: 支持 response API
    add_model("doubao-2.0-pro", "doubao-seed-2-0-pro-260215", 3.2, 16, "minimal")
    add_model("doubao-2.0-pro-think", "doubao-seed-2-0-pro-260215", 3.2, 16, "medium")
    add_model("doubao-2.0-pro-think-max", "doubao-seed-2-0-pro-260215", 3.2, 16, "high")
    add_model("doubao-2.0-lite", "doubao-seed-2-0-lite-260215", 0.6, 3.6, "minimal")
    add_model("doubao-2.0-lite-think", "doubao-seed-2-0-lite-260215", 0.6, 3.6, "medium")
    add_model("doubao-2.0-mini", "doubao-seed-2-0-mini-260215", 0.2, 2, "minimal")
    add_model("doubao-2.0-code", "doubao-seed-2-0-code-preview-260215", 3.2, 16, "medium")

    
    add_model("doubao-1.8", "doubao-seed-1-8-251228", 0.8, 8, "minimal")
    add_model("doubao-1.8-think", "doubao-seed-1-8-251228", 0.8, 8, "medium")
    
    add_model("doubao-1.8", "doubao-seed-1-8-251228", 0.8, 8, "minimal")
    add_model("doubao-1.8-think", "doubao-seed-1-8-251228", 0.8, 8, "medium")
    add_model("doubao-1.8-think-max", "doubao-seed-1-8-251228", 0.8, 8, "high")
    add_model("doubao-code", "doubao-seed-code-preview-251028",1.2, 8)
    add_model("doubao-1.6-flash", "doubao-seed-1-6-flash-250828", 0.3, 0.6, "minimal")
    add_model("deepseek-r1", "deepseek-r1-250528", 4, 16)
    add_model("deepseek-v3.2", "deepseek-v3-2-251201", 2, 3)
    
    # add_model("Doubao-thinking-pro", "doubao-1-5-thinking-pro-250415", 4, 16)
    # add_model("Doubao-vision-pro-32k", "doubao-1-5-vision-pro-32k-250115", 3, 9)
    # add_model("Doubao-pro-256k", "doubao-1-5-pro-256k-250115", 5, 9)
    
    