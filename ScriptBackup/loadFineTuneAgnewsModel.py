import transformers
import textattack

unwrapModel = transformers.BertForSequenceClassification.from_pretrained('Finetune-AGNEWS')
tokenizer = transformers.BertTokenizer.from_pretrained('Finetune-AGNEWS')
model = textattack.models.wrappers.HuggingFaceModelWrapper(unwrapModel, tokenizer)
