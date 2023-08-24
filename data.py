from torch.utils.data import Dataset
import transformers as t
import datasets as d


TEMPLATE_YES_INPUT = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
TEMPLATE_NOT_INPUT = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

class TrainDataset(Dataset):
  def __init__(self):
    self.tokenizer = t.AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    self.tokenizer.pad_token_id = 0
    self.tokenizer.padding_side = "left"
    self.ds = d.load_dataset("yahma/alpaca-cleaned")
    self.ds = self.ds["train"]
    self.ds = self.ds.map(self.prompt, remove_columns=["instruction", "input", "output"], load_from_cache_file=False, num_proc=8)
    self.ds = self.ds.map(self.tokenize, remove_columns=["prompt"], load_from_cache_file=False, num_proc=8)

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):
    return self.ds[idx]

  def prompt(self, elm):
    TEMPLATE = TEMPLATE_NOT_INPUT if not elm["input"] else TEMPLATE_YES_INPUT
    prompt = TEMPLATE.format(instruction=elm["instruction"], input=elm["input"])
    prompt = prompt + elm["output"]
    return {"prompt": prompt}

  def tokenize(self, elm):
    res = self.tokenizer(elm["prompt"])
    res["input_ids"].append(self.tokenizer.eos_token_id)
    res["attention_mask"].append(1)
    res["labels"] = res["input_ids"].copy()
    return res

  def max_seq_len(self):
    return max([len(elm["input_ids"]) for elm in self.ds])