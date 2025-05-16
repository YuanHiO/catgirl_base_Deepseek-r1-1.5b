from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# === 路径配置 ===
base_model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
lora_path = "deepseek-catgirl-fine-tuned"
output_dir = "./deepseek-catgirl-merged"

# === 加载 base 模型 ===
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# === 加载 LoRA Adapter 并合并 ===
model = PeftModel.from_pretrained(base_model, lora_path)
model = model.merge_and_unload()

# === 保存合并后的模型 ===
model.save_pretrained(output_dir)

# 同时保存 tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=True)
tokenizer.save_pretrained(output_dir)
