from typing import List, Optional
from openai import OpenAI
from jinja2 import Environment, FileSystemLoader

class VLLMServeInterface:
    def __init__(self, model_name: str, use_lora: bool = False, lora_name: Optional[str] = None, api_key: str = "Empty", port: int = 8000):
        """
        Initialize vLLM model interface

        Args:
            model_name: model name or path
            use_lora: whether to use LoRA
            lora_path: LoRA weight path (optional, usually specified by model name in serve mode)
        """
        self.model_name = lora_name if use_lora else model_name
        self.port = port
        self.client = OpenAI(
            api_key=api_key,
            base_url=f"http://localhost:{port}/v1",
        )

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9
    ) -> List[str]:
        """
        Generate text using vLLM model

        Args:
            prompt: input prompt
            max_length: maximum number of tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling probability

        Returns:
            List of generated text
        """
        env = Environment(loader=FileSystemLoader("."))
        template = env.get_template("utils/chat_template.jinja")
        if "gemma" in self.model_name:
            messages = [
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant in decision making."},
                {"role": "user", "content": prompt},
            ]
        prompt = template.render(messages=messages)
        chat_response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            stream=False,
            max_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
        )
        return chat_response.choices[0].text
        