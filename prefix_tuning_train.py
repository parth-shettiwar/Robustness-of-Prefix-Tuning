# -*- coding: utf-8 -*-




from datasets import load_dataset
from transformers import set_seed


randomSeed = 25
set_seed(randomSeed)
datas = "sst2"
lr = 6e-5
#snli, ag_news, sst2, imdb, yelp_polarity
prefix_len = 30
type_model = 1
#1 for prefix tuning and 0 for finetuning
# model saved in ./training_output

if(datas=="ag_news"):
  labs = 4
elif(datas=="snli"):
  labs = 3 
else:
  labs = 2  

if(datas=="sst2"):
  sent_lab = "sentence"
elif(datas=="snli"):
  sent_lab = "res"  
else:
  sent_lab = "text"  


if(type_model==1):
      model_pref = "prefix_"
else:
      model_pref = "finetune_"
if(type_model):      
  model_name = model_pref+datas+str(prefix_len) + "S" +  str(randomSeed)
else:
  model_name = model_pref+datas+ "S" + str(randomSeed)






if(datas=="yelp_polarity" or datas=="snli"):
  train_dataset = load_dataset(datas,split='train[2%:12%]')
  validation_dataset = load_dataset(datas,split='train[:2%]')
  test_dataset = load_dataset(datas,split='test')
else:
  train_dataset = load_dataset(datas,split='train[10%:90%]')
  validation_dataset = load_dataset(datas,split='train[90%:]+train[:10%]')
  test_dataset = load_dataset(datas,split='test')  




from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def encode_batch(batch):
  return tokenizer(batch[sent_lab], max_length=80, truncation=True, padding="max_length")


# dsd
if(datas=="snli"):
  res_train = [i + tokenizer.sep_token + j for i, j in zip(train_dataset["premise"], train_dataset["hypothesis"])]
  res_val = [i + tokenizer.sep_token + j for i, j in zip(validation_dataset["premise"], validation_dataset["hypothesis"])]
  res_test = [i + tokenizer.sep_token + j for i, j in zip(test_dataset["premise"], test_dataset["hypothesis"])]
  train_dataset = train_dataset.add_column("res", res_train)
  validation_dataset = validation_dataset.add_column("res", res_val)
  test_dataset = test_dataset.add_column("res", res_test)
  train_dataset = train_dataset.filter(lambda example: example["label"]>-1)
  validation_dataset = validation_dataset.filter(lambda example: example["label"]>-1)
  test_dataset = test_dataset.filter(lambda example: example["label"]>-1)

train_dataset = train_dataset.map(encode_batch, batched=True)
train_dataset.set_format(type="torch", columns=[sent_lab,"input_ids", "attention_mask", "label"])

validation_dataset = validation_dataset.map(encode_batch, batched=True)
validation_dataset.set_format(type="torch", columns=[sent_lab,"input_ids", "attention_mask", "label"])

test_dataset = test_dataset.map(encode_batch, batched=True)
test_dataset.set_format(type="torch", columns=[sent_lab,"input_ids", "attention_mask", "label"])



print(test_dataset)
print((train_dataset['label']))
print((test_dataset['label']))
print(min(test_dataset['label']))
print(max(test_dataset['label']))
print(min(train_dataset['label']))
print(max(train_dataset['label']))

import numpy as np
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction, Trainer
from transformers.adapters import PrefixTuningConfig
from transformers import  BertModel, AutoModelForSequenceClassification



def compute_accuracy(p: EvalPrediction):
      preds = np.argmax(p.predictions, axis=1)
      return {"acc": (preds == p.label_ids).mean()}

def training(type_model):
  config = AutoConfig.from_pretrained(
      "bert-base-uncased",
      num_labels=labs,
  )
  model = AutoModelForSequenceClassification.from_pretrained(
      "bert-base-uncased",
      config=config,
  )
  if(type_model==1):
    config = PrefixTuningConfig(flat=False, prefix_length=prefix_len)
    model.add_adapter("prefix_tuning", config=config)
    model.train_adapter("prefix_tuning")

    training_args = TrainingArguments(
        learning_rate=lr,
        num_train_epochs=50,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        logging_steps=200,
        save_strategy = "epoch",
        logging_strategy = "epoch",
        output_dir="./training_output/"+model_name,
        evaluation_strategy = "epoch",
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
        report_to="tensorboard"
    )

    

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_accuracy,
    )

  else:
    training_args = TrainingArguments(
    learning_rate=lr,
    num_train_epochs=50,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    logging_steps=200,
    logging_strategy = "epoch",
    save_strategy = "epoch",
    evaluation_strategy = "epoch",
    output_dir="./training_output/"+model_name,
    overwrite_output_dir=True,
    report_to="tensorboard")

 



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy,
    )
  return model, trainer


model, trainer = training(type_model)
trainer.train()
base = "./training_output/"
if(type_model):
  model.save_pretrained(base+model_name+"/model/")
  model.save_adapter(base+model_name+"/prefix/", 'prefix_tuning')
  trainer.save_model(base + model_name + "/saver/")
else:
  trainer.save_model(base+model_name)


