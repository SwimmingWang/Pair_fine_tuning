import argparse
import os
import sys
from accelerate import Accelerator

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dpo_trainer import MYDPOTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用DPO训练模型")

    # 模型相关参数
    parser.add_argument("--model_name", type=str, default="/data/models/gemma-2-2b-it", help="模型路径")
    
    # 训练参数
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="每个设备的训练批次大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1,
                        help="每个设备的评估批次大小")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="学习率")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="输入序列的最大长度")
    parser.add_argument("--max_prompt_length", type=int, default=2048,
                        help="提示词的最大长度")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="梯度累积步数")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="评估间隔步数")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="验证数据比例")
    parser.add_argument("--beta", type=float, default=0.2,
                        help="DPO温度参数")
    parser.add_argument("--val_batch_size", type=int, default=2, help="Validation batch size")
    parser.add_argument("--evaluation_steps", type=int, default=50, help="Evaluation interval in steps")

    # LoRA参数
    parser.add_argument("--use_lora_train_dpo", action="store_true",
                        help="使用LoRA进行DPO训练")
    parser.add_argument("--policy_adapter_path", type=str, default=None,
                        help="策略模型适配器路径")
    parser.add_argument("--ref_model_path", type=str, default=None,
                        help="参考模型路径")
    parser.add_argument("--ref_adapter_path", type=str, default=None,
                        help="参考模型适配器路径")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, 
                        default="q_proj, k_proj, v_proj, o_proj",
                        help="LoRA目标模块，用逗号分隔")

    # 数据和输出路径
    parser.add_argument("--dpo_data_path", type=str, required=True,
                        help="DPO数据文件路径")
    parser.add_argument("--template_path", type=str, required=True,
                        help="Jinja模板文件路径")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="保存最佳LoRA检查点的目录")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="保存检查点的间隔步数")

    # 日志参数
    parser.add_argument("--wandb_project", type=str, default="dpo-model-training",
                        help="Wandb项目名称")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb运行名称")
    parser.add_argument("--disable_wandb", action="store_true",
                        help="禁用Wandb日志记录")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="数据加载器工作进程数")

    args = parser.parse_args()
    
    # 设置随机种子
    import torch
    import random
    import numpy as np
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 初始化accelerator
    accelerator = Accelerator()
    
    # 创建训练器并开始训练
    trainer = MYDPOTrainer(args, accelerator)
    trainer.train()