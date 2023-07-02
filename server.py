import transformers as t
import gradio as gr
import peft as p
import torch
import time


TEMP = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
# NAME = "decapoda-research/llama-7b-hf"
# m = t.LlamaForCausalLM.from_pretrained(NAME, load_in_8bit=True, torch_dtype=torch.float16)
# config = p.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.005, bias="none", task_type="CAUSAL_LM")
# m = p.get_peft_model(m, config)
# tokenizer = t.LlamaTokenizer.from_pretrained(NAME)
# tokenizer.pad_token_id = 0
# adapters_weights = torch.load("./bessy/adapter_model.bin")
# p.set_peft_model_state_dict(m, adapters_weights)


g = t.AutoModelForCausalLM.from_pretrained('gpt2-xl').to("cuda:0")
g = torch.compile(g)
s = t.AutoTokenizer.from_pretrained('gpt2-xl')
s.pad_token_id = 0


# def generate(txt_prompt):
#   batch = tokenizer(txt_prompt, return_tensors="pt")
#   batch = {k: torch.tensor(v).unsqueeze(0).to("cuda:0") for k, v in batch.items()}
#   store = []
#   for _ in range(500):
#     token = torch.argmax(m(**batch).logits[:, -1, :], dim=-1)
#     batch["input_ids"] = torch.cat([batch["input_ids"], token.unsqueeze(1)], dim=-1)
#     batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(token).unsqueeze(1)], dim=-1)
#     store.append(token.item())
#     response = tokenizer.decode(store)
#     yield response
#     if token == tokenizer.eos_token_id: break


def generate(txt_prompt):
  print("PROMPT", txt_prompt)
  batch = s(txt_prompt, return_tensors="pt")
  batch = {k: torch.tensor(v).unsqueeze(0).to("cuda:0") for k, v in batch.items()}
  store = []
  for _ in range(50):
    print(g(**batch).logits.shape)
    token = torch.argmax(g(**batch).logits[:, :, -1, :], dim=-1)
    print("TOKEN", token)
    batch["input_ids"] = torch.cat([batch["input_ids"], token.unsqueeze(1)], dim=-1)
    batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(token).unsqueeze(1)], dim=-1)
    store.append(token.item())
    response = s.decode(store)
    yield response
    if token == s.eos_token_id: break


with gr.Blocks() as demo:
  chatbot = gr.Chatbot()
  msg = gr.Textbox()
  clear = gr.Button("Clear")

  def user(user_message, history):
    return "", history + [[user_message, None]]

  def bot(history):
    question = history[-1][0]
    prompt = TEMP.format(instruction=question)
    prompt = question
    for response in generate(prompt):
      history[-1][1] = response
      yield history

  msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
  clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch(server_name="0.0.0.0")