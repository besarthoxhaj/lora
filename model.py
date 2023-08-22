import transformers as t
import peft
import torch
import os


def get_model():
  NAME = "NousResearch/Llama-2-7b-hf"
  is_ddp = int(os.environ.get("WORLD_SIZE", 1)) != 1
  device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if is_ddp else None
  m = t.AutoModelForCausalLM.from_pretrained(NAME, load_in_8bit=True, torch_dtype=torch.float16, device_map=device_map)
  m = peft.prepare_model_for_kbit_training(m)
  config = peft.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.005, bias="none", task_type="CAUSAL_LM")
  m = peft.get_peft_model(m, config)
  return m
