import datasets
import textattack

hugDataset = datasets.load_from_disk('./Datasets/Imdb/test.hf')
dataset = textattack.datasets.HuggingFaceDataset(hugDataset)
