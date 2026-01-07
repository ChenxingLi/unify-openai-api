import os
from openai import AsyncOpenAI
from dataclasses import dataclass
from .openai import OpenAIProxy
from . import ChatCompletion
        
@dataclass
class QwenModel(OpenAIProxy):
    think: bool
    
    async def make_request(self, data: dict):
        data.setdefault("extra_body", dict())["enable_thinking"] = self.think
        data.setdefault("extra_body", dict())["enable_search"] = True
        return await super().make_request(data)
    
        
def regiester_models(models: dict[str, ChatCompletion]):
    client = AsyncOpenAI(api_key=os.getenv("ALIYUN_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    def add_model(name, target, input_price, output_price, think):
        model = QwenModel(client=client, name=name, target=target, input_price=input_price, output_price=output_price, think=think)
        models[model.model_id] = model
    
    add_model("qwen-plus", "qwen-plus-2025-04-28", 0.8, 2, False)
    add_model("qwen-plus-think", "qwen-plus-2025-04-28", 0.8, 16, True)
    add_model("qwq-plus", "qwq-plus-2025-03-05", 1.6, 4, True)
    add_model("qwen-max", "qwen-max-2025-01-25", 2.4, 9.6, False)
    
    