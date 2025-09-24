from typing import List, Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

class HuggingFaceInterface:
    def __init__(self, model_name: str, use_lora: bool = False, lora_path: Optional[str] = None):
        """
        Initialize the Hugging Face model interface
        
        Args:
            model_name: model name or path
            lora_path: path to LoRA weights (optional)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 加载LoRA权重
        if use_lora and lora_path is not None:
            print(f"Loading LoRA weights from: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print("LoRA weights loaded successfully!")
        
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9
    ) -> str:
        """
        Use the Hugging Face model to generate text
        
        Args:
            prompt: input prompt
            max_length: maximum length of the generated text
            temperature: sampling temperature
            top_p: nucleus sampling probability
            
        Returns:
            generated text string
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # 返回第一个生成的文本，与 vLLM Serve 接口保持一致
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)