import os
import anthropic
from llms.base import BaseLLM

class AnthropicLLM(BaseLLM):
    def __init__(self, api_key:str, model:str = "claude-2"):
        self.client = anthropic.Client(api_key=api_key)
        self.model = model
        
    def generate(self, prompt:str) -> str:
        response = self.client.completions.create(
            model=self.model,
            max_tokens_to_sample=500,
            temperature=0.7,
            system = "You are a helpful food recommendation assistant.",
            messages = [
                {"role": "user", "content": prompt}]
        )
        return response.completion        