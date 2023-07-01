from torch.utils.data import Dataset
import transformers as t
import datasets as d


TEMPLATE_YES_INPUT = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
TEMPLATE_NOT_INPUT = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"


class TrainData(Dataset):
  def __init__(self):
    self.tokenizer = t.LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    self.tokenizer.pad_token_id = (0)
    self.tokenizer.padding_side = "left"
    self.ds = d.load_dataset("yahma/alpaca-cleaned")
    self.ds = self.ds["train"].select(range(3))
    self.ds = self.ds.map(self.prompt)

  def __getitem__(self, idx):

    elms = self.ds[idx]
    print("elms", elms)
    # pmps = elms.map(self.prompt)

    print(f"__getitem__:{idx}", flush=True)
    item = {"foo": "bar"}
    return item

  def prompt(self, elm):

    if not elm["input"]:
      prompt = TEMPLATE_NOT_INPUT.format(instruction=elm["instruction"])
    else:
      prompt = TEMPLATE_YES_INPUT.format(instruction=elm["instruction"], input=elm["input"])

    return prompt + elm["output"]

  def tokenize(prompt, add_eos_token=True):
      # there's probably a way to do this with the tokenizer settings
      # but again, gotta move fast
      result = tokenizer(
          prompt,
          truncation=True,
          max_length=cutoff_len,
          padding=False,
          return_tensors=None,
      )
      if (
          result["input_ids"][-1] != tokenizer.eos_token_id
          and len(result["input_ids"]) < cutoff_len
          and add_eos_token
      ):
          result["input_ids"].append(tokenizer.eos_token_id)
          result["attention_mask"].append(1)

      result["labels"] = result["input_ids"].copy()

      return result