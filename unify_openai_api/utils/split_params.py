from typing import Any

def split_params(supported_fields: set[str], data: dict[str, Any]) -> tuple[dict, dict]:
    supported, unsupported = {}, {}
    for key, value in data.items():
        if key in supported_fields:
            supported[key] = value
        else:
            unsupported[key] = value
    return supported, unsupported