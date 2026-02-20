from dataclasses import dataclass
import os

from openai import AsyncOpenAI


from ..request_modifers.open_webui import OpenWebUIRequest
from ..request_modifers.interface import SetModel, RequestModifier
from ..response_handlers.cost_record import ChatCompletionCostRecord

from ..backends.openai import OpenAIProxy


from ..types.llm_api import LLMApi
        
@dataclass
class QwenModifier(RequestModifier):
    think: bool
    
    def modify_data(self, data: dict):
        data.setdefault("extra_body", dict())["enable_thinking"] = self.think
        # data.setdefault("extra_body", dict())["enable_search"] = True
        return data
    
        
def regiester_models(models: dict[str, LLMApi]):
    client = AsyncOpenAI(api_key=os.getenv("ALIYUN_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    def qwen_model(target, input_price, output_price, think):
        request_modifiers = [
            SetModel(model_name=target),
            QwenModifier(think=think),
            OpenWebUIRequest()
        ]
        response_handlers = [
            ChatCompletionCostRecord(model_id=target, input_price=input_price/7, output_price=output_price/7),
        ]
        return OpenAIProxy(client=client, request_modifiers=request_modifiers, response_handlers=response_handlers)
    
    models["qwen-plus"] = qwen_model("qwen-plus-2025-04-28", input_price=0.8, output_price=2, think=False)
    models["qwen-plus-think"] = qwen_model("qwen-plus-2025-04-28", input_price=0.8, output_price=16, think=True)
    models["qwq-plus"] = qwen_model("qwq-plus-2025-03-05", input_price=1.6, output_price=4, think=True)
    models["qwen-max"] = qwen_model("qwen-max-2025-01-25", input_price=2.4, output_price=9.6, think=False)
    