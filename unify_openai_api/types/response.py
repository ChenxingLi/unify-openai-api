from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class ApiResponse:
    response: Any
    user_id: Optional[str]
    stream: bool