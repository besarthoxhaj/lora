# %%
import transformers as t
import datasets as d
import torch
import peft as p
import utils as u
# # %%
# NAME = "decapoda-research/llama-7b-hf"
# args = { "load_in_8bit": True, "torch_dtype": torch.float16 }
# model = t.AutoModelForCausalLM.from_pretrained(NAME, **args) # device_map="auto"
# print("00.model", model)
# # %%
# tokenizer = t.LlamaTokenizer.from_pretrained(NAME)
# tokenizer.pad_token_id = (0)
# tokenizer.padding_side = "left"
# # %%
# config = p.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
# model = p.get_peft_model(model, config)
# model.print_trainable_parameters()
# # %%
# print("01.model", model)
# # %%
# text = "The French revolution lasted from"
# batch = tokenizer(text, return_tensors="pt", add_special_tokens=False)
# batch = {k: v.to("cuda:0") for k, v in batch.items()}
# print("02.batch", batch)
# # %%
# logits, loss = model(**batch)
# print("03.logits", logits)
# print("04.loss", loss)

# %%
print("bes")
ds = d.load_dataset("yahma/alpaca-cleaned")
ds = ds["train"].select(range(2)).map(u.gen_tokenize_prompt)

# %%
