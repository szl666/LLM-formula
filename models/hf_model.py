
import os
import torch
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_model import BaseLLMModel

class HuggingFaceModel(BaseLLMModel):
    """Hugging Face model implementation"""
    
    def __init__(self, model_name: str, device=None, dtype=None, cache_dir=None, **kwargs):
        super().__init__(model_name, **kwargs)
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)
        
        # Model parameters
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 0.9)
        self.num_beams = kwargs.get("num_beams", 1)
        self.max_new_tokens = kwargs.get("max_new_tokens", 256)
        self.min_new_tokens = kwargs.get("min_new_tokens", 0)
        
        # Load model and tokenizer
        token = os.environ.get("HF_TOKEN", None)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, torch_dtype=self.dtype, cache_dir=cache_dir, token=token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.dtype, cache_dir=cache_dir, 
            token=token, device_map='auto'
        )
        self.model.eval()
        
        # Configure tokenizer
        if kwargs.get("tokenizer_pad"):
            self.tokenizer.pad_token = kwargs["tokenizer_pad"]
        if kwargs.get("tokenizer_padding_side"):
            self.tokenizer.padding_side = kwargs["tokenizer_padding_side"]
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        temperature = kwargs.get("temperature", self.temperature)
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        
        # Process prompt
        if '<image>' in prompt:
            prompt = prompt.replace('<image>', '')
        
        messages = self._get_messages(prompt)
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs, do_sample=True, temperature=temperature,
            top_k=self.top_k, top_p=self.top_p, num_beams=self.num_beams,
            max_new_tokens=max_new_tokens, min_new_tokens=self.min_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode
        outputs = outputs[0][len(inputs[0]):]
        decoded_output = self.tokenizer.decode(outputs, skip_special_tokens=True)
        
        # Clean output
        decoded_output = decoded_output.replace("assistant", "").replace("user", "").replace("system", "")
        return decoded_output.strip()
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat-based generation"""
        # Convert messages to prompt format
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        return self.generate(prompt, **kwargs)
    
    def _get_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Convert prompt to messages format"""
        return [{"role": "user", "content": prompt}]
