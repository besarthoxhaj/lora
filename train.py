import transformers
import model
import data
import os


is_ddp = int(os.environ.get("WORLD_SIZE", 1)) != 1
m = model.get_model()
ds = data.TrainDataset()
collator = transformers.DataCollatorForSeq2Seq(ds.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)


# import torch
# import peft
# adapters_weights = torch.load("./output/checkpoint-800/adapter_model.bin")
# peft.set_peft_model_state_dict(m, adapters_weights)

title = str(input("Title: "))

num_train_epochs = 1
learning_rate = 3e-4
optim="adamw_torch"
batch_size = 4

wandb.init(
  project="Llama",
  name= title,
  config={
    "epochs": num_train_epochs,
    "learning_rate": learning_rate,
    "optim": optim,
    "batch_size": batch_size,
  }
)


trainer = transformers.Trainer(
  model=m,
  train_dataset=ds,
  data_collator=collator,
  args=transformers.TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="no",
    save_strategy="steps",
    eval_steps=None,
    save_steps=200,
    output_dir="./output",
    save_total_limit=3,
    report_to="wandb",
    ddp_find_unused_parameters=False if is_ddp else None,
  ),
)


m.config.use_cache = False
trainer.train()
m.save_pretrained("./weights")
