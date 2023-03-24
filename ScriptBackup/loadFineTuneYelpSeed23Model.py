import transformers
import textattack

unwrapModel = transformers.BertForSequenceClassification.from_pretrained('Finetune-Yelp-Seed23')
tokenizer = transformers.BertTokenizer.from_pretrained('Finetune-Yelp-Seed23')
model = textattack.models.wrappers.HuggingFaceModelWrapper(unwrapModel, tokenizer)
