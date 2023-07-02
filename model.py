import transformers as t
import peft as p
import torch


def get_model():
  NAME = "decapoda-research/llama-7b-hf"
  m = t.LlamaForCausalLM.from_pretrained(NAME, load_in_8bit=True, torch_dtype=torch.float16)
  m = p.prepare_model_for_int8_training(m)
  config = p.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.005, bias="none", task_type="CAUSAL_LM")
  m = p.get_peft_model(m, config)
  m.print_trainable_parameters()
  return m


def generate(m, prompt):
  tokenizer = t.LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
  batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
  batch = {k: v.to("cuda:0") for k, v in batch.items()}
  print("batch", batch["input_ids"])
  # for _ in range(45):
  #   token = torch.argmax(m(**batch).logits[:, -1, :], dim=-1)
  #   batch["input_ids"] = torch.cat([batch["input_ids"], token.unsqueeze(1)], dim=-1)
  #   batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(token).unsqueeze(1)], dim=-1)
  #   print(tokenizer.decode(batch["input_ids"].squeeze().tolist()))