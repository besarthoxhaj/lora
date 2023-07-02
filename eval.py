# %%
import transformers as t
import peft as p
import model
import torch
#%%
def generate(m, prompt):
  tokenizer = t.LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
  tokenizer.pad_token_id = 0
  batch = tokenizer(prompt, add_special_tokens=False)
  batch["input_ids"].insert(0, tokenizer.eos_token_id)
  batch["attention_mask"].insert(0, 1)
  batch = {k: torch.tensor(v).unsqueeze(0).to("cuda:0") for k, v in batch.items()}
  print("batch", batch)
  for _ in range(450):
    token = torch.argmax(m(**batch).logits[:, -1, :], dim=-1)
    if token[0].item() == tokenizer.eos_token_id: break
    batch["input_ids"] = torch.cat([batch["input_ids"], token.unsqueeze(1)], dim=-1)
    batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(token).unsqueeze(1)], dim=-1)
    print(tokenizer.decode(batch["input_ids"].squeeze().tolist()))
# %%
m = model.get_model()
#%%
TEMPLATE = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
prompt = TEMPLATE.format(instruction="Python how to insert something at the beginning of a list?")
# %%
generate(m, prompt)
# %%
adapters_weights = torch.load("./bessy/adapter_model.bin")
p.set_peft_model_state_dict(m, adapters_weights)
# %%
print(m)
# %%
generate(m, prompt)