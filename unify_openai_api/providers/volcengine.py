import os
from typing import Optional
from openai import AsyncOpenAI
from dataclasses import dataclass

from ..request_modifers.open_webui import OpenWebUIRequest
from ..request_modifers.interface import SetModel, RequestModifier
from ..response_handlers.cost_record import ChatCompletionCostRecord

from ..backends.openai import OpenAIProxy

from ..types.llm_api import LLMApi


@dataclass
class DoubaoModifier(RequestModifier):
    reasoning_effort: Optional[str]

    def modify_data(self, data: dict):
        if self.reasoning_effort is not None:
            data["reasoning_effort"] = self.reasoning_effort
        return data


def regiester_models(models: dict[str, LLMApi]):
    client = AsyncOpenAI(api_key=os.getenv("VOLC_API_KEY"), base_url="https://ark.cn-beijing.volces.com/api/v3")

    def doubao_model(target, input_price, output_price, reasoning_effort=None):
        request_modifiers = [
            SetModel(model_name=target),
            OpenWebUIRequest()
        ]

        if reasoning_effort is None:
            request_modifiers.append(DoubaoModifier(reasoning_effort=reasoning_effort))

        response_handlers = [
            ChatCompletionCostRecord(model_id=target, input_price=input_price/7, output_price=output_price/7),
        ]
        return OpenAIProxy(client=client, request_modifiers=request_modifiers, response_handlers=response_handlers)

    # TODO: 支持豆包动态定价长度
    # TODO: 支持 response API
    models["doubao-2.0-pro"] =doubao_model("doubao-seed-2-0-pro-260215", 3.2, 16, "minimal")
    models["doubao-2.0-pro-think"] = doubao_model("doubao-seed-2-0-pro-260215", 3.2, 16, "medium")
    models["doubao-2.0-pro-think-max"] = doubao_model("doubao-seed-2-0-pro-260215", 3.2, 16, "high")
    models["doubao-2.0-lite"] = doubao_model("doubao-seed-2-0-lite-260215",0.6, 3.6, "minimal")
    models["doubao-2.0-lite-think"] = doubao_model("doubao-seed-2-0-lite-260215", 0.6, 3.6, "medium")
    models["doubao-2.0-mini"] = doubao_model("doubao-seed-2-0-mini-260215", 0.2, 2, "minimal")
    models["doubao-2.0-code"] = doubao_model("doubao-seed-2-0-code-preview-260215", 3.2, 16, "medium")

    models["doubao-1.8"] = doubao_model("doubao-seed-1-8-251228", 0.8, 8, "minimal")
    models["doubao-1.8-think"] = doubao_model("doubao-seed-1-8-251228", 0.8, 8, "medium")
    models["doubao-1.8-think-max"] = doubao_model("doubao-seed-1-8-251228", 0.8, 8, "high")
    models["doubao-code"] = doubao_model("doubao-seed-code-preview-251028", 1.2, 8)
    models["doubao-1.6-flash"] = doubao_model("doubao-seed-1-6-flash-250828", 0.3, 0.6, "minimal")

    models["deepseek-r1"] = doubao_model("deepseek-r1-250528", 4, 16)
    models["deepseek-v3.2"] = doubao_model("deepseek-v3-2-251201", 2, 3)

    # add_model("Doubao-thinking-pro", "doubao-1-5-thinking-pro-250415", 4, 16)
    # add_model("Doubao-vision-pro-32k", "doubao-1-5-vision-pro-32k-250115", 3, 9)
    # add_model("Doubao-pro-256k", "doubao-1-5-pro-256k-250115", 5, 9)
