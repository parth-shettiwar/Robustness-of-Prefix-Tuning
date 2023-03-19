



from datasets import load_dataset
from transformers import set_seed


randomSeed = 24
set_seed(randomSeed)
datas = "sst2"
prefix_len = 30
type_model = 0

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
print(test_dataset["label"])


from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
from transformers.adapters import AutoAdapterModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def encode_batch(batch):
  return tokenizer(batch[sent_lab], max_length=80, truncation=True, padding="max_length")


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

test_dataset = test_dataset.map(encode_batch, batched=True)
test_dataset.set_format(type="torch", columns=[sent_lab,"input_ids", "attention_mask", "label"])

validation_dataset = validation_dataset.map(encode_batch, batched=True)
validation_dataset.set_format(type="torch", columns=[sent_lab,"input_ids", "attention_mask", "label"])


import numpy as np
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction, Trainer
from transformers.adapters import PrefixTuningConfig
from transformers import  BertModel, AutoModelForSequenceClassification


print(test_dataset)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

def compute_accuracy(p: EvalPrediction):
      preds = np.argmax(p.predictions, axis=1)
      return {"acc": (preds == p.label_ids).mean()}

def training(type_model):
  config = AutoConfig.from_pretrained(
      "bert-base-uncased",
      num_labels=labs,
  )

  
  if(type_model==1):
   
    base = "/home/parth/sota/GRAN/training_output"
    model_name = "/prefix_imdb30S25/"
    check_num = "2650"
    base2 = base+model_name+"checkpoint-"+check_num+"/prefix_tuning/"

    model = AutoAdapterModel.from_pretrained(base+model_name+"/model")
    adapter_name = model.load_adapter(base2) 
    model.set_active_adapters(adapter_name)
   
    

    training_args = TrainingArguments(
        learning_rate=5e-5,
        num_train_epochs=50,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        logging_steps=50,
        output_dir="./training_output",
        overwrite_output_dir=True,
        remove_unused_columns=False,
        report_to="tensorboard"
    )

    

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_accuracy,
    )

  else:
    base = "/home/parth/sota/GRAN/training_output"
    model_name = "/finetune_imdbS25/"
    check_num = "1027"
    base2 = base+model_name+"checkpoint-"+check_num+"/pytorch_model.bin"
    print(base2)

    print("$$$$$$$$$$$$$$$$$")

    model = AutoModelForSequenceClassification.from_pretrained(
        base2,
        config=config,
    )
    training_args = TrainingArguments(
    learning_rate=5e-5,
    num_train_epochs=50,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    logging_steps=50,
    output_dir="./training_output",
    overwrite_output_dir=True,
    report_to="tensorboard")





    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy,
    )
  return trainer


trainer = training(type_model)

print(trainer.evaluate())

