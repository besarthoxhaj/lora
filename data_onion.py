from torch.utils.data import Dataset
import transformers as t
import datasets as d
import pandas as pd


TEMPLATE = "Below is a title for an article. Write an article that appropriately suits the title: \n\n### Title:\n{title}\n\n### Article:\n"

class OnionTrainDataset(Dataset):
  def __init__(self):
    self.tokenizer = t.AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    self.tokenizer.pad_token_id = 0
    self.tokenizer.padding_side = "left"

    self.data_df = pd.read_csv("onion.csv")
    self.data_df = self.data_df.drop(columns=["Published Time"])

    self.ds_prompts = self.data_df.apply(self.prompt,  axis=1)
    self.ds = self.ds_prompts.apply(self.tokenize, axis=1)

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):
    return self.ds.iloc[idx]

  def prompt(self, elm):
    prompt = TEMPLATE.format(title=elm["Title"])
    prompt = prompt + str(elm["Content"])
    return {"prompt": prompt}

  def tokenize(self, elm):
    res = self.tokenizer(elm["prompt"])
    res["input_ids"].append(self.tokenizer.eos_token_id)
    res["attention_mask"].append(1)
    res["labels"] = res["input_ids"].copy()
    return res

  def max_seq_len(self):
    return max([len(elm["input_ids"]) for elm in self.ds])
  
OnionTrainDataset()