import transformers
import textattack
from transformers.adapters import BertAdapterModel

unwrapModel = BertAdapterModel.from_pretrained('./Prefix-imdb/prefix_tuning')
tokenizer = transformers.BertTokenizer.from_pretrained('Prefix-imdb')
unwrapModel.load_adapter('Prefix-imdb')
unwrapModel.set_active_adapters('prefix-tuning')
model = textattack.models.wrappers.HuggingFaceModelWrapper(unwrapModel, tokenizer)
