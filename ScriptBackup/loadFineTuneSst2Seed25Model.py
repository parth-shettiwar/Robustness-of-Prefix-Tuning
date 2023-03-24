import transformers
import textattack

unwrapModel = transformers.BertForSequenceClassification.from_pretrained('Finetune-SST2-Seed25')
tokenizer = transformers.BertTokenizer.from_pretrained('Finetune-SST2-Seed25')
model = textattack.models.wrappers.HuggingFaceModelWrapper(unwrapModel, tokenizer)
