import logging
logging.basicConfig(
        level=logging.DEBUG,  # 设置最低输出级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        # handlers=[
        #     logging.FileHandler("app.log"),  # 输出到文件
        #     logging.StreamHandler()         # 输出到控制台
        # ]
    )

import logging
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager

from unify_openai_api.types.llm_api import LLMApi, AppState, get_typed_state
from unify_openai_api.usage_db.writer import AsyncDBWriter

import unify_openai_api.providers.aliyun
import unify_openai_api.providers.volcengine
import unify_openai_api.providers.deerapi

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ 这里是原 startup 函数的内容
    models: dict[str, LLMApi] = dict()
    writer = AsyncDBWriter()
    writer.start()

    unify_openai_api.providers.aliyun.regiester_models(models)
    unify_openai_api.providers.volcengine.regiester_models(models)
    unify_openai_api.providers.deerapi.regiester_models(models)

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
            } for model in get_typed_state(request).models
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    代理处理 OpenAI 的 chat.completions 请求，支持 stream 和 non-stream 模式。
    """
    state = get_typed_state(request)
    models = state.models
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

    model: LLMApi = models[model_id]

    response = await model.make_request(data)
    return await model.handle_response(response, state)

# 启动服务
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
