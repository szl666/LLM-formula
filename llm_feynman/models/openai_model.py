
from typing import Dict, Any, Optional, List
from openai import OpenAI
from .base_model import BaseLLMModel

class OpenAIModel(BaseLLMModel):
    """OpenAI model implementation with support for O3 API"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None, base_url: str = None, **kwargs):
        super().__init__(model_name, **kwargs)
        
        # Initialize OpenAI client with custom base_url support for O3 API
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        self.temperature = kwargs.get("temperature", 0.2)
        self.max_tokens = kwargs.get("max_tokens", 2048)
        self.top_p = kwargs.get("top_p", 1.0)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in mathematical formula discovery and scientific analysis."},
            {"role": "user", "content": prompt}
        ]
        return self.chat(messages, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat-based generation"""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=kwargs.get("top_p", self.top_p)
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            import traceback
            traceback.print_exc()
            return ""
