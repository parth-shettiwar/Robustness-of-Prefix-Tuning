import transformers
import textattack

unwrapModel = transformers.BertForSequenceClassification.from_pretrained('FineTune-yelp')
tokenizer = transformers.BertTokenizer.from_pretrained('FineTune-yelp')
model = textattack.models.wrappers.HuggingFaceModelWrapper(unwrapModel, tokenizer)
