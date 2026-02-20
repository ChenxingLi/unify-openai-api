from typing import Any, Dict, List, Optional

from ..types.state import AppState


class ResponseHandler:
    def handle_response(self, state: AppState, user_id: str, data: Any) -> Any:
        """子类重写此方法来修改 data。默认不修改。"""
        return data
    
    def handle_response_frame(self, state: AppState, user_id: str, frame: Any) -> Any:
        """子类重写此方法来修改 data。默认不修改。"""
        return frame
    
def handle_response(handlers: List[ResponseHandler], state: AppState, user_id: str, data: Any) -> Any:
    """依次应用 handlers 来修改 data。"""
    for handler in handlers:
        data = handler.handle_response(state, user_id, data)
    return data

def handle_response_frames(handlers: List[ResponseHandler], state: AppState, user_id: str, frame: Any) -> Optional[Any]:
    """依次应用 handlers 来修改 frames。"""
    for handler in handlers:
        frame = handler.handle_response_frame(state, user_id, frame)
        if frame is None:
            return None
    return frame