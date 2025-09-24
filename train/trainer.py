import os
import math
import json
import logging
from typing import List, Dict, Optional, Iterable

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training
)
from tqdm import tqdm
import random
import numpy as np

# ======================
# 日志
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# 实用函数
# ======================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_best_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif torch.cuda.is_available():
        return torch.float16
    else:
        return torch.float32

def count_trainable_parameters(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} "
                f"({100 * trainable / total:.2f}%)")

def module_names(model: nn.Module) -> Iterable[str]:
    for n, _ in model.named_modules():
        yield n

def guess_lora_target_modules(model: nn.Module):
    """
    依据常见 CausalLM 架构自动猜测 LoRA 注入模块名称。
    你也可以直接在 config 中显式传入 target_modules 覆盖这里的结果。
    """
    names = list(module_names(model))
    candidates = []
    # 常见投影层
    common = ["q_proj", "k_proj", "v_proj", "o_proj"]
    # 也兼容一些实现把投影写作 "W_pack"/"wo"/"wqkv"
    alt = ["wo", "wq", "wk", "wv", "wqkv", "W_pack"]
    for n in names:
        if any(n.endswith(f".{c}") or n == c for c in common + alt):
            pieces = n.split(".")
            candidates.append(pieces[-1])

    # 兜底：如果没发现，则回退到最常见的四个
    if not candidates:
        candidates = ["q_proj", "k_proj", "v_proj", "o_proj"]
    # 去重，保留相对稳定的顺序
    uniq = []
    for c in candidates:
        if c not in uniq:
            uniq.append(c)
    logger.info(f"自动识别到的 LoRA target_modules: {uniq}")
    return uniq

# ======================
# 单样本 SFT 数据集（原版）
# ======================
class SFTDataset(Dataset):
    """
    支持两种模板：
    1) 自定义字符串模板 (默认)："### Human: ... \\n### Assistant: ..."
    2) tokenizer.apply_chat_template，通过 use_chat_template=True 开启
    仅训练回答部分；padding 与 prompt 位置在 labels 中设为 -100
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        prompt_template: str = "### Human: {instruction}\n### Assistant: {response}",
        use_chat_template: bool = False,
        add_eos_to_answer: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.use_chat_template = use_chat_template
        self.add_eos_to_answer = add_eos_to_answer
        self.data = self.load_data(data_path)

    def load_data(self, data_path: str) -> List[Dict]:
        with open(data_path, "r", encoding="utf-8") as f:
            if data_path.endswith(".json"):
                data = json.load(f)
            elif data_path.endswith(".jsonl"):
                data = [json.loads(line) for line in f]
            else:
                raise ValueError("不支持的文件格式，请使用 .json 或 .jsonl")
        logger.info(f"加载了 {len(data)} 条训练数据")
        return data

    def _build_texts(self, item: Dict):
        # 标准字段对齐
        if "instruction" in item and "response" in item:
            ins, ans = item["instruction"], item["response"]
        elif "input" in item and "output" in item:
            ins, ans = item["input"], item["output"]
        elif "text" in item:
            return "", item["text"]
        else:
            raise ValueError(f"不支持的数据格式: {item.keys()}")

        if self.use_chat_template:
            msgs_prompt = [{"role": "user", "content": ins}]
            msgs_full = [
                {"role": "user", "content": ins},
                {"role": "assistant", "content": ans},
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                msgs_prompt, tokenize=False, add_generation_prompt=True
            )
            full_text = self.tokenizer.apply_chat_template(
                msgs_full, tokenize=False, add_generation_prompt=False
            )
        else:
            prompt_text = self.prompt_template.format(instruction=ins, response="")
            full_text = self.prompt_template.format(instruction=ins, response=ans)

        if self.add_eos_to_answer and self.tokenizer.eos_token and not full_text.endswith(self.tokenizer.eos_token):
            full_text = full_text + self.tokenizer.eos_token

        return prompt_text, full_text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt_text, full_text = self._build_texts(item)

        # 先对 prompt 单独编码以获得其 token 长度
        prompt_ids = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False
        )["input_ids"]

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        labels[attention_mask == 0] = -100
        prompt_len = min(len(prompt_ids), self.max_length)
        if prompt_len > 0:
            labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# ======================
# 成对样本数据集：同题x + (extra_a, y_a), (extra_b, y_b)
# ======================
class PairedSFTDataset(Dataset):
    """
    每个样本同时返回 a、b 两个 (prompt, full_text) 的编码。
    字段要求：
      - instruction: str
      - extra_a: str, response_a: str
      - extra_b: str, response_b: str
    模板规则：
      - 若 use_chat_template=True：把 (extra + instruction) 作为user消息，(response) 作为assistant消息
      - 否则：使用 prompt_template，把 extra 融到 instruction 前（可按需调整）
    训练时：仅计算回答部分的loss（prompt部分labels=-100）。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        prompt_template: str = "### Human: {instruction}\n### Assistant: {response}",
        use_chat_template: bool = False,
        add_eos_to_answer: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.use_chat_template = use_chat_template
        self.add_eos_to_answer = add_eos_to_answer
        self.data = self._load_data(data_path)
        logger.info(f"[Paired] 加载了 {len(self.data)} 条训练数据")

    def _load_data(self, data_path: str) -> List[Dict]:
        with open(data_path, "r", encoding="utf-8") as f:
            if data_path.endswith(".jsonl"):
                return [json.loads(line) for line in f]
            elif data_path.endswith(".json"):
                return json.load(f)
            else:
                raise ValueError("PairedSFTDataset 仅支持 .json / .jsonl")

    def _build_one(self, instruction: str, extra: str, answer: str):
        # 你可以自定义：把 extra 拼接到 instruction 前面
        merged_ins = f"{extra}\n{instruction}" if extra else instruction

        if self.use_chat_template:
            msgs_prompt = [{"role": "user", "content": merged_ins}]
            msgs_full = [
                {"role": "user", "content": merged_ins},
                {"role": "assistant", "content": answer},
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                msgs_prompt, tokenize=False, add_generation_prompt=True
            )
            full_text = self.tokenizer.apply_chat_template(
                msgs_full, tokenize=False, add_generation_prompt=False
            )
        else:
            prompt_text = self.prompt_template.format(instruction=merged_ins, response="")
            full_text = self.prompt_template.format(instruction=merged_ins, response=answer)

        if self.add_eos_to_answer and self.tokenizer.eos_token and not full_text.endswith(self.tokenizer.eos_token):
            full_text = full_text + self.tokenizer.eos_token

        return prompt_text, full_text

    def _encode_pair(self, prompt_text: str, full_text: str):
        # prompt 的 token 长度，用于把对应 labels 置为 -100
        prompt_ids = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False
        )["input_ids"]

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        prompt_len = min(len(prompt_ids), self.max_length)
        if prompt_len > 0:
            labels[:prompt_len] = -100

        return input_ids, attention_mask, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 字段名严格匹配（若你有不同字段名，在此处改）
        ins = item["instruction"]
        extra_a, ans_a = item["extra_a"], item["response_a"]
        extra_b, ans_b = item["extra_b"], item["response_b"]

        pa, fa = self._build_one(ins, extra_a, ans_a)
        pb, fb = self._build_one(ins, extra_b, ans_b)

        ids_a, mask_a, labels_a = self._encode_pair(pa, fa)
        ids_b, mask_b, labels_b = self._encode_pair(pb, fb)

        return {
            "input_ids_a": ids_a,
            "attention_mask_a": mask_a,
            "labels_a": labels_a,
            "input_ids_b": ids_b,
            "attention_mask_b": mask_b,
            "labels_b": labels_b,
        }

# ======================
# LoRA 训练器（加入 paired 支持 + 从训练集切分验证集）
# ======================
class LoraTrainer:
    def __init__(
        self,
        model_name: str,
        train_data_path: str,
        output_dir: str,
        max_length: int = 512,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        save_steps: int = 500,
        eval_steps: int = 500,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        val_data_path: Optional[str] = None,
        use_chat_template: bool = False,
        grad_checkpointing: bool = False,
        seed: int = 42,
        # —— LoRA 相关 —— #
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        # —— （可选）低比特加载 —— #
        use_qlora_4bit: bool = False,
        bnb_4bit_compute_dtype: Optional[torch.dtype] = None,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        load_in_8bit: bool = False,
        # —— 成对训练开关与权重 —— #
        paired: bool = False,
        loss_weight_a: float = 1.0,
        loss_weight_b: float = 1.0,
        # —— 新增：从训练集中切分验证集 —— #
        val_from_train: bool = True,
        val_ratio: float = 0.1,
    ):
        set_seed(seed)

        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.use_chat_template = use_chat_template
        self.grad_checkpointing = grad_checkpointing

        # LoRA
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules

        # 成对训练
        self.paired = paired
        self.loss_weight_a = loss_weight_a
        self.loss_weight_b = loss_weight_b

        # 数据切分
        self.val_from_train = val_from_train
        self.val_ratio = max(0.0, min(0.5, val_ratio))  # 安全范围

        os.makedirs(output_dir, exist_ok=True)

        # 设备 & dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = get_best_dtype()
        logger.info(f"使用设备: {self.device}, dtype: {self.dtype}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        from transformers import BitsAndBytesConfig
        quantization_config = None
        model_kwargs = dict(trust_remote_code=True)
        if use_qlora_4bit or load_in_8bit:
            if use_qlora_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype or self.dtype
                )
                logger.info("使用 4-bit 量化（QLoRA）加载模型")
            elif load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("使用 8-bit 量化加载模型")

            model_kwargs.update(dict(
                quantization_config=quantization_config,
                device_map="auto"
            ))
        else:
            model_kwargs.update(dict(
                torch_dtype=self.dtype,
            ))

        # Base Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        if use_qlora_4bit or load_in_8bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=grad_checkpointing
            )
        else:
            self.model.to(self.device)
            if grad_checkpointing:
                self.model.gradient_checkpointing_enable()

        self.model.config.use_cache = False 

        if self.lora_target_modules is None:
            self.lora_target_modules = guess_lora_target_modules(self.model)

        # 注入 LoRA
        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.lora_target_modules,
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()
        count_trainable_parameters(self.model)
        num_workers = 4

        if self.paired:
            full_dataset = PairedSFTDataset(
                train_data_path, self.tokenizer, max_length,
                use_chat_template=use_chat_template
            )
        else:
            full_dataset = SFTDataset(
                train_data_path, self.tokenizer, max_length,
                use_chat_template=use_chat_template
            )

        # 优先：独立验证集
        if val_data_path:
            if self.paired:
                self.val_dataset = PairedSFTDataset(
                    val_data_path, self.tokenizer, max_length,
                    use_chat_template=use_chat_template
                )
            else:
                self.val_dataset = SFTDataset(
                    val_data_path, self.tokenizer, max_length,
                    use_chat_template=use_chat_template
                )
            self.train_dataset = full_dataset
            logger.info("使用独立 val_data_path 作为验证集")
        # 其次：从训练集切分
        elif self.val_from_train and self.val_ratio > 0.0 and len(full_dataset) >= 2:
            val_size = max(1, int(round(len(full_dataset) * self.val_ratio)))
            train_size = len(full_dataset) - val_size
            g = torch.Generator()
            g.manual_seed(seed)
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size], generator=g)
            logger.info(f"从训练集中切分验证集: train={train_size}, val={val_size} (ratio={self.val_ratio:.2f})")
        else:
            self.train_dataset = full_dataset
            self.val_dataset = None
            if self.eval_steps:
                logger.info("未创建验证集（eval_steps>0 但无验证集时将跳过评估）")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if torch.cuda.is_available() and num_workers > 0 else False
        )

        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True if torch.cuda.is_available() and num_workers > 0 else False
            )
        else:
            self.val_loader = None

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        total_update_steps = math.ceil(len(self.train_loader) * num_epochs / self.gradient_accumulation_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_update_steps
        )

        self.use_amp = (self.dtype == torch.float16 and torch.cuda.is_available())
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.global_step = 0
        self.best_val_loss = float("inf")

        logger.info(f"总 batch 步数: {len(self.train_loader) * self.num_epochs}, 预计优化步数: {total_update_steps}")

        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

    def compute_loss(self, batch):
        if self.model.device.type == "cuda" and not any(
            getattr(p, "is_quantized", False) for p in self.model.parameters()
        ):
            dev = self.device
        else:
            dev = next(self.model.parameters()).device

        if "input_ids_a" in batch and "input_ids_b" in batch:
            input_ids_a = batch["input_ids_a"].to(dev, non_blocking=True)
            attention_mask_a = batch["attention_mask_a"].to(dev, non_blocking=True)
            labels_a = batch["labels_a"].to(dev, non_blocking=True)

            input_ids_b = batch["input_ids_b"].to(dev, non_blocking=True)
            attention_mask_b = batch["attention_mask_b"].to(dev, non_blocking=True)
            labels_b = batch["labels_b"].to(dev, non_blocking=True)

            out_a = self.model(input_ids=input_ids_a, attention_mask=attention_mask_a, labels=labels_a)
            out_b = self.model(input_ids=input_ids_b, attention_mask=attention_mask_b, labels=labels_b)
            loss = self.loss_weight_a * out_a.loss + self.loss_weight_b * out_b.loss
            return loss

        input_ids = batch["input_ids"].to(dev, non_blocking=True)
        attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
        labels = batch["labels"].to(dev, non_blocking=True)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss

    def train_step(self, batch):
        self.model.train()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss = self.compute_loss(batch)
            loss = loss / self.gradient_accumulation_steps

        self.scaler.scale(loss).backward()
        return loss.item()

    def evaluate(self):
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss = self.compute_loss(batch)
                total_loss += float(loss.item())
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        logger.info(f"验证损失: {avg_loss:.4f}")
        return avg_loss

    # —— 保存 / 导出 —— #
    def save_adapter(self, save_path: str):
        logger.info(f"保存 LoRA 适配器到: {save_path}")
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # 也保存优化器/调度器等训练状态（可选）
        torch.save({
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss
        }, os.path.join(save_path, "training_state.pt"))

    def merge_and_save_full_model(self, base_model_name_or_path: str, adapter_path: str, merged_path: str):
        """
        将 LoRA 权重合并进 Base 模型并导出全量权重（推理无需 PEFT）。
        注意：合并会把权重真正写入并增大文件体积；仅在需要独立部署时使用。
        """
        logger.info(f"合并 LoRA 到基座模型并导出: {merged_path}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, torch_dtype=self.dtype, trust_remote_code=True
        )
        merged = PeftModel.from_pretrained(base, adapter_path)
        merged = merged.merge_and_unload()  # 把 LoRA 合并回权重
        os.makedirs(merged_path, exist_ok=True)
        merged.save_pretrained(merged_path)
        self.tokenizer.save_pretrained(merged_path)
        logger.info("合并完成。")

    def train(self):
        logger.info("start LoRA training...")

        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.num_epochs):
            logger.info(f"开始第 {epoch + 1}/{self.num_epochs} 轮训练")
            epoch_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            for step, batch in enumerate(progress_bar):
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.max_grad_norm
                    )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

                    self.global_step += 1

                    progress_bar.set_postfix({
                        "loss": f"{loss:.4f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })

                    if self.eval_steps and self.global_step % self.eval_steps == 0:
                        val_loss = self.evaluate()
                        if val_loss is not None and val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                        self.model.train()

                    if self.save_steps and self.global_step % self.save_steps == 0:
                        ckpt_dir = os.path.join(self.output_dir, f"adapter-{self.global_step}")
                        self.save_adapter(ckpt_dir)

            avg_epoch_loss = epoch_loss / max(1, num_batches)
            logger.info(f"第 {epoch + 1} 轮平均损失: {avg_epoch_loss:.4f}")

        final_dir = os.path.join(self.output_dir, "final_adapter")
        self.save_adapter(final_dir)
        logger.info("LoRA 训练完成！")

class LoraSFTInference:
    def __init__(self, base_or_merged_path: str, adapter_path: Optional[str] = None,
                 max_length: int = 512, use_chat_template: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.use_chat_template = use_chat_template

        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path or base_or_merged_path, trust_remote_code=True
        )
        if adapter_path:
            base = AutoModelForCausalLM.from_pretrained(
                base_or_merged_path,
                torch_dtype=get_best_dtype(),
                trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(base, adapter_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_or_merged_path, torch_dtype=get_best_dtype(), trust_remote_code=True
            )

        self.model.to(self.device)
        self.model.eval()
        self.model.config.use_cache = True

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(
        self,
        instruction: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        if self.use_chat_template:
            msgs = [{"role": "user", "content": instruction}]
            prompt = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"### Human: {instruction}\n### Assistant:"

        max_input_tokens = max(8, self.max_length - max_new_tokens)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
            add_special_tokens=False
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if self.use_chat_template:
            return full[len(prompt):].strip()
        else:
            if "### Assistant:" in full:
                return full.split("### Assistant:")[-1].strip()
            return full[len(prompt):].strip()
