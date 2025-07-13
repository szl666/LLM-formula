
import openai
from typing import Dict, Any, Optional, List
from .base_model import BaseLLMModel

class OpenAIModel(BaseLLMModel):
    """OpenAI model implementation"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None, **kwargs):
        super().__init__(model_name, **kwargs)
        
        if api_key:
            openai.api_key = api_key
        
        self.temperature = kwargs.get("temperature", 1.0)
        self.max_tokens = kwargs.get("max_tokens", 256)
        self.top_p = kwargs.get("top_p", 1.0)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat-based generation"""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=kwargs.get("top_p", self.top_p)
        )
        
        return response.choices[0].message.content.strip()
