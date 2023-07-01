import transformers as t


NAME = "decapoda-research/llama-7b-hf"
tokenizer = t.LlamaTokenizer.from_pretrained(NAME)
tokenizer.pad_token_id = (0)
tokenizer.padding_side = "left"


PROMPT_YES_INPUT = """
Below is an instruction that describes a task, paired with an input
that provides further context. Write a response that appropriately
completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


PROMPT_NOT_INPUT = """
Below is an instruction that describes a task. Write a response that
appropriately completes the request.

### Instruction: {instruction}\### Response:
"""


def gen_tokenize_prompt(elm):
  print("elm", elm)
  if elm["input"] is "":
    prompt = PROMPT_NOT_INPUT.format(instruction=elm["instruction"])
  else:
    prompt = PROMPT_YES_INPUT.format(instruction=elm["instruction"], input=elm["input"])

  res = prompt + elm["output"]
  print("res", res)

