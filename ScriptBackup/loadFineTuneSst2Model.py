import transformers
import textattack

unwrapModel = transformers.BertForSequenceClassification.from_pretrained('Finetune-SST2')
tokenizer = transformers.BertTokenizer.from_pretrained('Finetune-SST2')
model = textattack.models.wrappers.HuggingFaceModelWrapper(unwrapModel, tokenizer)
