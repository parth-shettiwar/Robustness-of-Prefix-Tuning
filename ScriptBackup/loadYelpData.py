import datasets
import textattack

hugDataset = datasets.load_from_disk('./Datasets/Yelp/test.hf')
dataset = textattack.datasets.HuggingFaceDataset(hugDataset)
