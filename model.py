import transformers as t
import peft as p
import torch


class Model():
  def __init__(self):
    NAME = "decapoda-research/llama-7b-hf"
    m = t.LlamaForCausalLM.from_pretrained(NAME, load_in_8bit=True, torch_dtype=torch.float16)
    m = p.prepare_model_for_int8_training(m)
    config = p.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.005, bias="none", task_type="CAUSAL_LM")
    m = p.get_peft_model(m, config)
    m.print_trainable_parameters()
    self.m = m


# # %%
# import transformers as t
# import torch
# # %%
# NAME = "decapoda-research/llama-13b-hf"
# tokenizer = t.LlamaTokenizer.from_pretrained(NAME)
# model = t.AutoModelForCausalLM.from_pretrained(NAME) # device_map="auto"
# # %%
# model.half().to("cuda:0")
# # %%
# text = "The French revolution lasted from"
# batch = tokenizer(text, return_tensors="pt", add_special_tokens=False)
# batch = {k: v.to("cuda:0") for k, v in batch.items()}
# for _ in range(12):
#     token = torch.argmax(model(**batch).logits[:, -1, :], dim=-1)
#     batch["input_ids"] = torch.cat([batch["input_ids"], token.unsqueeze(1)], dim=-1)
#     batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(token).unsqueeze(1)], dim=-1)
#     print(tokenizer.decode(batch["input_ids"].squeeze().tolist()))
# # %%
