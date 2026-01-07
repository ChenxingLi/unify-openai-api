import logging
logging.basicConfig(
        level=logging.DEBUG,  # 设置最低输出级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        # handlers=[
        #     logging.FileHandler("app.log"),  # 输出到文件
        #     logging.StreamHandler()         # 输出到控制台
        # ]
    )


import unify_openai_api.backends.deerapi
import unify_openai_api.backends.volcengine
import unify_openai_api.backends.aliyun
import os
import json
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from openai._exceptions import OpenAIError
from contextlib import asynccontextmanager

from unify_openai_api.openai_utils import split_openai_params
from unify_openai_api.backends import ChatCompletion
from unify_openai_api.usage_db.writer import AsyncDBWriter


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ 这里是原 startup 函数的内容
    models: dict[str, ChatCompletion] = dict()
    writer = AsyncDBWriter()
    writer.start()

    unify_openai_api.backends.aliyun.regiester_models(models)
    unify_openai_api.backends.volcengine.regiester_models(models)
    unify_openai_api.backends.deerapi.regiester_models(models)

    app.state.models = models
    app.state.writer = writer

    yield

    writer.stop()

app = FastAPI(lifespan=lifespan)  # 关键：传递 lifespan


@app.get("/v1/models")
async def list_models(request: Request):
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
            } for model in request.app.state.models
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    代理处理 OpenAI 的 chat.completions 请求，支持 stream 和 non-stream 模式。
    """
    models = request.app.state.models
    try:
        data = await request.json()
        logger.debug(f"request: {data}")
        model_id = data["model"]
    except Exception as e:
        logger.error(f"Error parsing request JSON: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if model_id not in models:
        raise HTTPException(
            status_code=400, detail=f"Unknown Model {model_id}")

    model: ChatCompletion = models[model_id]

    response = await model.make_request(data)
    return await model.handle_response(response, request.app.state)

# 启动服务
if __name__ == "__main__":
    import uvicorn

    

    uvicorn.run(app, host="0.0.0.0", port=8000)
