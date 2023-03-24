import transformers
import textattack

unwrapModel = transformers.BertForSequenceClassification.from_pretrained('FineTune-AGNEWS-Seed24')
tokenizer = transformers.BertTokenizer.from_pretrained('FineTune-AGNEWS-Seed24')
model = textattack.models.wrappers.HuggingFaceModelWrapper(unwrapModel, tokenizer)
