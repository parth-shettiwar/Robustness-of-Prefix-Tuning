import datasets
import textattack

hugDataset = datasets.load_from_disk('./Datasets/ag_news/test.hf')
dataset = textattack.datasets.HuggingFaceDataset(hugDataset)
