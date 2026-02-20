from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional

ModifierCallable = Callable[[Dict[str, Any]], Dict[str, Any]]

# 中间件基类：所有自定义中间件继承此类
class RequestModifier:
    def modify_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """子类重写此方法来修改 data。默认不修改。"""
        return data

@dataclass
class SetModel(RequestModifier):
    model_name: str
    
    def modify_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["model"] = self.model_name
        return data

def modify_request(modifiers: List[RequestModifier], data: Dict[str, Any]) -> Dict[str, Any]:
    """依次应用 modifiers 来修改 data。"""
    for modifier in modifiers:
        data = modifier.modify_data(data)
    return data


# def concat_request_modifiers(modifiers: List[RequestModifier]) -> ModifierCallable:
#     """构建链式修改函数：从内到外包装。"""
#     modifier = lambda data: data # 从基础修改函数开始（可以是 identity 函数）
#     for mw in reversed(modifiers):  # 逆序：最内层先包装
#         modifier = _wrap_modifier(mw, modifier)
#     return modifier

# def _wrap_modifier(modifier: RequestModifier, next_modifier: ModifierCallable) -> ModifierCallable:
#     """构建包装函数：返回一个新的 modifier callable，用于链式组成。"""
#     def _middleware(data: Dict[str, Any]) -> Dict[str, Any]:
#         # 先调用当前中间件的修改
#         modified_data = modifier.modify_data(data)
#         # 然后调用下一个中间件
#         return next_modifier(modified_data)
    
#     return _middleware