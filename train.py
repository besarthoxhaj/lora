# %%
import transformers as t
import peft as p
import torch
import model
import data


m = model.get_model()
ds = data.TrainDataset()


# adapters_weights = torch.load("./output/checkpoint-800/adapter_model.bin")
# p.set_peft_model_state_dict(m, adapters_weights)


trainer = t.Trainer(
  model=m,
  train_dataset=ds,
  args=t.TrainingArguments(
    per_device_train_batch_size=16,
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
  ),
  data_collator=t.DataCollatorForSeq2Seq(
    ds.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
  ),
)


trainer.train()
m.save_pretrained("./bessy")
