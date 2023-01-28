# -*- coding: utf-8 -*-




from datasets import load_dataset

# train_dataset = load_dataset("imdb",split='train')
# test_dataset_sun = load_dataset("imdb",split='test[:100]')
# test_dataset = load_dataset("imdb",split='test')

train_dataset = load_dataset("yelp_polarity",split='train')
test_dataset_sun = load_dataset("yelp_polarity",split='test[:100]')
test_dataset = load_dataset("yelp_polarity",split='test')

# dataset = load_dataset("imdb")

# train_dataset.num_rows

# test_dataset

# test_dataset.num_rows

# test_dataset_sun.num_rows

from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def encode_batch(batch):
  return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

train_dataset = train_dataset.map(encode_batch, batched=True)
train_dataset.set_format(type="torch", columns=['text',"input_ids", "attention_mask", "label"])

test_dataset = test_dataset.map(encode_batch, batched=True)
test_dataset.set_format(type="torch", columns=['text',"input_ids", "attention_mask", "label"])

test_dataset_sun = test_dataset_sun.map(encode_batch, batched=True)
test_dataset_sun.set_format(type="torch", columns=['text',"input_ids", "attention_mask", "label"])

test_dataset['label']

import numpy as np
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction, Trainer
from transformers.adapters import PrefixTuningConfig
from transformers import  BertModel, AutoModelForSequenceClassification


model_name = "prefix-yelp"

def compute_accuracy(p: EvalPrediction):
      preds = np.argmax(p.predictions, axis=1)
      return {"acc": (preds == p.label_ids).mean()}

def training(type_model):
  config = AutoConfig.from_pretrained(
      "bert-base-uncased",
      num_labels=2,
  )
  model = AutoModelForSequenceClassification.from_pretrained(
      "bert-base-uncased",
      config=config,
  )
  if(type_model==1):
    config = PrefixTuningConfig(flat=False, prefix_length=20)
    model.add_adapter("prefix_tuning", config=config)
    model.train_adapter("prefix_tuning")

    training_args = TrainingArguments(
        learning_rate=3e-5,
        num_train_epochs=50,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        logging_steps=200,
        save_strategy = "epoch",
        output_dir="./training_output/"+model_name,
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
        report_to="tensorboard",
        evaluation_strategy="epoch"
    )

    

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_accuracy,
    )
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #       print(name)
    # print(model)      

  else:
    training_args = TrainingArguments(
    learning_rate=1e-5,
    num_train_epochs=50,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    logging_steps=200,
    save_strategy = "epoch",
    output_dir="./training_output/"+model_name,
    overwrite_output_dir=True,
    report_to="tensorboard",
    evaluation_strategy="epoch")

    # from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy,
    )
    for name, param in model.named_parameters():
        if param.requires_grad:
          print(name)
    print(model) 
  return trainer

# %load_ext tensorboard
# %tensorboard --logdir "/content/training_output/runs"
trainer = training(1)
trainer.train()
base = "/home/parth/sota/GRAN/models/"
trainer.save_model(base+model_name)


# !ls -lh final_adapter