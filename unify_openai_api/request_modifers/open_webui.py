from dataclasses import dataclass

from .interface import RequestModifier


@dataclass
class OpenWebUIRequest(RequestModifier):
    """OpenWebUI 专用请求修改器：处理特定于 OpenWebUI 的请求字段。"""
    chat_completion_request: bool = True
    
    def modify_data(self, data: dict) -> dict:
        # 处理 openwebui_middleware 字段
        openwebui_meta = data.pop("openwebui_middleware", None)
        
        if openwebui_meta is not None:
            data["user_id"] = openwebui_meta.get("user_id", None)
        else:
            data["user_id"] = None

        if self.chat_completion_request and data.get("stream", False):
            data.setdefault("stream_options", dict())["include_usage"] = True
        
        return data