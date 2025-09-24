from typing import List, Optional
from vllm import LLM, SamplingParams
from jinja2 import Environment, FileSystemLoader

class VLLMInterface:
    def __init__(self, model_name: str, use_lora: bool = False, lora_path: Optional[str] = None):
        """
        Initialize vLLM model interface

        Args:
            model_name: model name or path
            use_lora: whether to use LoRA
            lora_path: LoRA weight path (optional, usually specified by model name in serve mode)
        """
        self.model_name = lora_path if use_lora else model_name
        self.llm = LLM(
            model=model_name,
            enforce_eager=True,
            # gpu_memory_utilization=0.5,
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
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_length, top_p=top_p)
        env = Environment(loader=FileSystemLoader("."))
        template = env.get_template("utils/chat_template.jinja")
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = template.render(messages=messages)
        outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text
