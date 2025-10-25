
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class BaseLLMModel(ABC):
    """Abstract base class for LLM models"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat-based generation"""
        pass
