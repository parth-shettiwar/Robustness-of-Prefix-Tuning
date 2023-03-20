# Robustness-of-Prefix-Tuning
This work has been done as part of the capstone project at University of California, Los Angeles
Team Members:  
* Parth Shettiwar
* Rohit Sunchu 
* Jakob Didio
  
    
Work exploring the robustness of natural language models to attack is greatly increasing as
models become commonly deployed in real world contexts where they are vulnerable to
malefactors. Because of this, it is of interest to
researchers to assess the effect that modifications to existing natural language models have
on their resilience to adversarial attack. Prefix
tuning has grown in popularity as an effective
alternative to standard fine-tuning of natural
language models, with numerous space and
flexibility benefits. But there has not yet been a
study to determine the effect that utilizing prefix tuning has on a models’ resistance to attack.
In this work, we perform a comprehensive empirical study of the adversarial robustness of
BERT tuned with a standard fine-tuning process, and with prefix tuning. 

## Codebase
The following libraries shoud be installed before running the following codes (can use pip):
* datasets
* transformers  

The codebase consists of the following files:
* `prefix_tuning_train.py` : This file is used to train the models using fine-tuning and prefix-tuning mainly.  
The various hyperparameters and usage has been commented inside the file. Need to set the following parameters
  * `randomSeed` : can be any integer value
  * `datas` : can be "snli", "ag_news", "sst2", "imdb", "yelp_polarity"
  * `lr` : sets the learning rate for training
  * `prefix_len` : determines the prefix length for prefix tuning training of the model. 
  * `type_model` : 0 for fine-tuning and 1 for prefix-tuning  
  
  Simply run the code as       
  ```
  python prefix_tuning_train.py
  ```
* `prefix_tuning_test.py`: This file is used to test the models using fine-tuning and prefix-tuning mainly. 
  Again set the parameters datas and type_model as mentioned above. 
  Also set the path of the checkpoint of the model in the `base`, `model_name`, `check_num` parameters appropriately. 
  
  Simply run the code as       
  ```
  python prefix_tuning_test.py
  ```


## Attack Examples

We have included an example workflow for attacking a trained BERT model prefix tuned on AG News. First, ensure that you have installed the [Textattack](https://github.com/QData/TextAttack) package along with the above mentioned packages. Then, ensure that the directory `./Datasets/ag_news/` contains a file `test.hf` which contains the examples to be attacked in standard Huggingface dataset format. Finally, ensure there is a directory `./Prefix-AGNEWS-Seed23/` that contains a saved Huggingface [bert-base-uncased](https://huggingface.co/bert-base-uncased) model in `Prefix-AGNEWS-Seed23/model/` and a saved adapterhub prefix tuning adapter in `Prefix-AGNEWS-Seed23/prefix_tuning/`. You may then run
```
source bertAttackCommandAgnewsPrefixSeed23.sh
```
which will commence the attack, saving logs and checkpoints in the current directory.
