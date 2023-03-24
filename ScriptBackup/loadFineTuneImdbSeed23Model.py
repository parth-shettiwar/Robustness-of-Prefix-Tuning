import transformers
import textattack

unwrapModel = transformers.BertForSequenceClassification.from_pretrained('FineTune-imdb-Seed23')
tokenizer = transformers.BertTokenizer.from_pretrained('FineTune-imdb-Seed23')
model = textattack.models.wrappers.HuggingFaceModelWrapper(unwrapModel, tokenizer)
