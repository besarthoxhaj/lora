#%%
import transformers
import torch
import peft
import time
#%%
tokenizer = transformers.LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
tokenizer.pad_token_id = 0
#%%
def generate(m, prompt):
  batch = tokenizer(prompt, add_special_tokens=False)
  batch["input_ids"].insert(0, tokenizer.eos_token_id)
  batch["attention_mask"].insert(0, 1)
  batch = {k: torch.tensor(v).unsqueeze(0).to("cuda:0") for k, v in batch.items()}

  m.eval()

  with torch.no_grad():
    for _ in range(100):
        token = torch.argmax(m(**batch).logits[:, -1, :], dim=-1)
        if token[0].item() == tokenizer.eos_token_id: break
        batch["input_ids"] = torch.cat([batch["input_ids"], token.unsqueeze(1)], dim=-1)
        batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(token).unsqueeze(1)], dim=-1)
  return tokenizer.decode(batch["input_ids"].squeeze().tolist())
# %%
m = transformers.LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", load_in_8bit=True, torch_dtype=torch.float16)
config = peft.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.005, bias="none", task_type="CAUSAL_LM")
m = peft.get_peft_model(m, config)
adapters_weights = torch.load("/home/fsuser/lora/weights/adapter_model.bin")
peft.set_peft_model_state_dict(m, adapters_weights)
#%%
TEMPLATE = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
prompt = TEMPLATE.format(instruction="Python how to insert something at the beginning of a list?")
start = time.time()
output = generate(m, prompt)
end = time.time()
print("Time taken: ", end - start)
print("Output:", output)
# %%
