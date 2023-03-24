import torch
import textattack
from transformers import BertTokenizer
from transformers.adapters import BertAdapterModel, PrefixTuningConfig

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

unwrapModel = BertAdapterModel.from_pretrained('./Prefix-AGNEWS/model/') 

unwrapModel.load_adapter('./Prefix-AGNEWS/prefix_tuning')

unwrapModel.set_active_adapters('prefix_tuning')

model = textattack.models.wrappers.HuggingFaceModelWrapper(unwrapModel, tokenizer)
