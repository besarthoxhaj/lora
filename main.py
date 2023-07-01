# %%
import transformers as t
import torch
# %%
NAME = "decapoda-research/llama-13b-hf"
tokenizer = t.LlamaTokenizer.from_pretrained(NAME)
model = t.AutoModelForCausalLM.from_pretrained(NAME) # device_map="auto"
# %%
model.half().to("cuda:0")
# %%
text = "The French revolution lasted from"
batch = tokenizer(text, return_tensors="pt", add_special_tokens=False)
batch = {k: v.to("cuda:0") for k, v in batch.items()}
for _ in range(12):
    token = torch.argmax(model(**batch).logits[:, -1, :], dim=-1)
    batch["input_ids"] = torch.cat([batch["input_ids"], token.unsqueeze(1)], dim=-1)
    batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(token).unsqueeze(1)], dim=-1)
    print(tokenizer.decode(batch["input_ids"].squeeze().tolist()))
# %%
