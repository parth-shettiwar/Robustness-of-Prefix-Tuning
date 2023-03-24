'''import transformers
import textattack

unwrapModel = transformers.BertForSequenceClassification.from_pretrained('FineTune-imdb')
tokenizer = transformers.BertTokenizer.from_pretrained('FineTune-imdb')
model = textattack.models.wrappers.HuggingFaceModelWrapper(unwrapModel, tokenizer)'''

import torch
import textattack
from transformers import BertTokenizer
from transformers.adapters import BertAdapterModel, PrefixTuningConfig

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

unwrapModel = BertAdapterModel.from_pretrained('./Prefix-imdb-new/model/') 

unwrapModel.load_adapter('./Prefix-imdb-new/checkpoint-2212/prefix_tuning')

unwrapModel.set_active_adapters('prefix_tuning')

model = textattack.models.wrappers.HuggingFaceModelWrapper(unwrapModel, tokenizer)
