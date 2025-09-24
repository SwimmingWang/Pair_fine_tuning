import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import LoraTrainer

def main():
    # ====== 配置 ======
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./models/lora_output")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_chat_template", type=bool, default=False)
    parser.add_argument("--grad_checkpointing", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=list, default=None)

    parser.add_argument("--use_qlora_4bit", type=bool, default=False)
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", type=bool, default=True)
    parser.add_argument("--load_in_8bit", type=bool, default=False)

    parser.add_argument("--train_type", type=str, default="sft", choices=["sft", "dpo", "pair"])
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--loss_weight_a", type=float, default=1.0)
    parser.add_argument("--loss_weight_b", type=float, default=1.0)

    parser.add_argument("--val_from_train", type=bool, default=True)
    parser.add_argument("--val_ratio", type=float, default=0.01)
    args = parser.parse_args()

    config = vars(args)

    data_path = config["train_data_path"]
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if not os.path.exists(data_path):
        if config["paired"]:
            raise ValueError("Paired dataset is not found.")
        else:
            raise ValueError("Single sample dataset is not found.")

    trainer = LoraTrainer(
        model_name=config["model_name"],
        train_data_path=config["train_data_path"],
        val_data_path=config["val_data_path"],
        output_dir=config["output_dir"],
        max_length=config["max_length"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        num_epochs=config["num_epochs"],
        warmup_steps=config["warmup_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        max_grad_norm=config["max_grad_norm"],
        use_chat_template=config["use_chat_template"],
        grad_checkpointing=config["grad_checkpointing"],
        seed=config["seed"],
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        lora_target_modules=config["lora_target_modules"],
        use_qlora_4bit=config["use_qlora_4bit"],
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
        load_in_8bit=config["load_in_8bit"],
        paired=config["paired"],
        loss_weight_a=config["loss_weight_a"],
        loss_weight_b=config["loss_weight_b"],
        val_from_train=config["val_from_train"],
        val_ratio=config["val_ratio"],
    )

    trainer.train()

if __name__ == "__main__":
    main()