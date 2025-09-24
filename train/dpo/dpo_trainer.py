import os
from functools import partial
import json
from typing import Dict, Any
import torch
import wandb
from datasets import Dataset
from jinja2 import Environment, FileSystemLoader
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from trl import DPOConfig, DPOTrainer, get_kbit_device_map
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM

from data import DPODataset


os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MYDPOTrainer:
    def __init__(self, args, accelerator):
        self.args = args
        self.accelerator = accelerator

        if accelerator.is_main_process:
            self._init_wandb()
        
        self._setup_tokenizer()
        self._setup_dataset()
        self._create_quantization_config()
        self._setup_models()

        self._setup_dpo_trainer()

        def save_model(self, output_dir: str, _internal_call: bool = False):
            self.model.save_pretrained(output_dir)
            self.processing_class.save_pretrained(output_dir)
            print(f"Saved PEFT model to {output_dir}")

        self.dpo_trainer.save_model = save_model.__get__(
            self.dpo_trainer, type(self.dpo_trainer)
        )

    def _init_wandb(self):
        wandb.init(
            project=self.args.wandb_project,
            name=self.args.wandb_run_name,
            config={k: v for k, v in vars(self.args).items() if isinstance(v, (int, float, str))}
        )

    def _setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name, padding_side="left"
        )
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("[PAD]")

    def _setup_dataset(self):
        # Load template
        env = Environment(
            loader=FileSystemLoader("/".join(self.args.template_path.split("/")[:-1]))
        )
        template = env.get_template(self.args.template_path.split("/")[-1])
        
        dataset = DPODataset(
            data_path=self.args.dpo_data_path,
            tokenizer=self.tokenizer,
            template=template,
            max_length=self.args.max_length,
        )

        generator = torch.Generator().manual_seed(42)
        val_ratio = getattr(self.args, "val_ratio", 0.05)
        train_size = min(int(len(dataset) * (1 - val_ratio)), len(dataset) - 2)
        val_size = len(dataset) - train_size
        
        # 分割数据集
        train_subset, val_subset = random_split(
            dataset, [train_size, val_size], generator=generator
        )
        
        # 将Subset转换为Dataset对象，以兼容TRL库
        def subset_to_list(subset):
            return [subset.dataset[i] for i in subset.indices]

        self.train_dataset = Dataset.from_list(subset_to_list(train_subset))
        self.val_dataset = Dataset.from_list(subset_to_list(val_subset))
        
        print(f"Dataset split: {len(self.train_dataset)} train, {len(self.val_dataset)} validation")

    def _create_quantization_config(self):
        # self.quant_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # )
        self.quant_config = None

    def _setup_models(self):
        # Setup policy model (the model being trained)
        if getattr(self.args, 'use_lora_train_dpo', False):
            print("Loading base model with LoRA for DPO training")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype="auto",
                quantization_config=self.quant_config if self.quant_config else None,
                device_map=get_kbit_device_map() if self.quant_config else None,
            ) 
            # Load LoRA adapter if specified
            if hasattr(self.args, 'policy_adapter_path') and self.args.policy_adapter_path:
                print(f"Loading policy adapter from {self.args.policy_adapter_path}")
                self.model = PeftModelForCausalLM.from_pretrained(
                    base_model,
                    self.args.policy_adapter_path,
                    is_trainable=True,
                    adapter_name="policy_adapter",
                )
            else:
                # Apply LoRA configuration
                peft_config = LoraConfig(
                    r=getattr(self.args, 'lora_r', 16),
                    lora_alpha=getattr(self.args, 'lora_alpha', 32),
                    lora_dropout=getattr(self.args, 'lora_dropout', 0.1),
                    target_modules=getattr(self.args, 'target_modules', 'q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj').split(","),
                    task_type="CAUSAL_LM",
                )
                self.model = get_peft_model(base_model, peft_config)
        else:
            print("Loading full model for DPO training")
            if hasattr(self.args, 'resume_from_checkpoint') and self.args.resume_from_checkpoint:
                print(f"Resuming full model from checkpoint: {self.args.resume_from_checkpoint}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.args.resume_from_checkpoint,
                    torch_dtype="auto",
                    quantization_config=self.quant_config if self.quant_config else None,
                    device_map=get_kbit_device_map() if self.quant_config else None,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    torch_dtype="auto",
                    quantization_config=self.quant_config,
                    device_map=get_kbit_device_map() if self.quant_config else None,
                )

        # Setup reference model (frozen)
        if getattr(self.args, 'ref_model_path', None):
            print(f"Loading reference model from {self.args.ref_model_path}")
            ref_base_model = AutoModelForCausalLM.from_pretrained(
                self.args.ref_model_path,
                torch_dtype="auto",
                quantization_config=self.quant_config if self.quant_config else None,
                device_map=get_kbit_device_map() if self.quant_config else None,
            )
            # Load LoRA adapter for reference model if specified
            if hasattr(self.args, 'ref_adapter_path') and self.args.ref_adapter_path:
                print(f"Loading reference model LoRA adapter from {self.args.ref_adapter_path}")
                self.ref_model = PeftModelForCausalLM.from_pretrained(
                    ref_base_model,
                    self.args.ref_adapter_path,
                    is_trainable=False,  # Reference model should be frozen
                    adapter_name="ref_adapter",
                )
            else:
                self.ref_model = ref_base_model
        else:
            print("Using same model as reference (will be frozen)")
            self.ref_model = None

        # Set pad token id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if self.ref_model:
            self.ref_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def _setup_dpo_trainer(self):
        # Calculate effective batch size
        num_processes = self.accelerator.num_processes
        global_batch_size = (
            self.args.per_device_train_batch_size
            * num_processes
            * self.args.gradient_accumulation_steps
        )
        
        print(f"Global batch size: {global_batch_size}")

        # Setup DPO training arguments
        training_args = DPOConfig(
            output_dir=self.args.checkpoint_dir,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            beta=self.args.beta,
            evaluation_strategy="steps",
            ddp_find_unused_parameters=True,
            max_prompt_length=self.args.max_prompt_length,
            max_length=self.args.max_length,
            logging_steps=1,
            save_steps=self.args.save_steps,
            eval_steps=self.args.evaluation_steps,
            optim="paged_adamw_8bit" if self.quant_config else "adamw_torch",
            bf16=True,
            report_to="wandb",
            disable_dropout=True,
        )

        # Initialize DPO trainer
        self.dpo_trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.tokenizer,
        )
        
        print("DPO trainer setup complete")

    def train(self):
        try:
            print("Starting DPO training...")
            train_stats = self.dpo_trainer.train()
            if self.accelerator.is_main_process:
                print("Saving final model checkpoint...")
                self.dpo_trainer.save_model(self.args.checkpoint_dir)
            return train_stats
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise