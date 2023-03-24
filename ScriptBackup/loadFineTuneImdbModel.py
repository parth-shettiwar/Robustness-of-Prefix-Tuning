import transformers
import textattack

unwrapModel = transformers.BertForSequenceClassification.from_pretrained('FineTune-imdb')
tokenizer = transformers.BertTokenizer.from_pretrained('FineTune-imdb')
model = textattack.models.wrappers.HuggingFaceModelWrapper(unwrapModel, tokenizer)
