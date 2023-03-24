import transformers
import textattack

unwrapModel = transformers.BertForSequenceClassification.from_pretrained('Finetune-Yelp-Seed24')
tokenizer = transformers.BertTokenizer.from_pretrained('Finetune-Yelp-Seed24')
model = textattack.models.wrappers.HuggingFaceModelWrapper(unwrapModel, tokenizer)
