import wandb
wb_token = ""

wandb.login(key=wb_token)
wandb.init(
    project="deepseek-catgirl",
    name="unsloth_lora_run",
    config={
        "r": 256,
        "lora_alpha": 256,
        "lora_dropout": 0.0,
        "learning_rate": 2e-4,
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "max_steps": 250
    }
)