from typing import Protocol, abstractmethod
from abc import ABC

class ChatCompletion(ABC):
    @property
    @abstractmethod
    def model_id(self):
        raise NotImplementedError
    
    @abstractmethod
    async def make_request(self, data: dict): ...
    
    @abstractmethod
    async def handle_response(self, obj, state): ...