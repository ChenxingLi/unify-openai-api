from .interface import RequestModifier

from anthropic.types import MessageParam, ThinkingConfigEnabledParam, ThinkingConfigDisabledParam

class OpenAIToAnthropicMiddleware(RequestModifier):
    """将 OpenAI LLMApi 请求转换为 Anthropic 格式的中间件。"""
    
    def modify_data(self, data: dict) -> dict:
        # 将 messages 转换为 Anthropic 的 MessageParam，如果需要
        data["messages"] = [MessageParam(**msg) for msg in data.pop("messages", [])]

        # Anthropic 使用 'max_tokens' 而非 'max_completion_tokens'，但支持 'max_tokens'
        if "max_completion_tokens" in data:
            data["max_tokens"] = data.pop("max_completion_tokens")
        
        # Anthropic 要求必须指定 max_tokens
        data.setdefault("max_tokens", 32768)
            
        # 添加 budger tokens
        if "budget_tokens" in data:
            budget_tokens = data.pop("budget_tokens")
            if budget_tokens > 0:
                data["thinking"] = ThinkingConfigEnabledParam(budget_tokens=budget_tokens, type = "enabled")
            else:
                data["thinking"] = ThinkingConfigDisabledParam(type = "disabled")
                
        if "reasoning_effort" in data:
            # TODO
            pass
        
        return data
    