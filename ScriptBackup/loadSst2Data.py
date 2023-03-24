import datasets
import textattack

hugDataset = datasets.load_from_disk('./Datasets/sst2/test.hf')
dataset = textattack.datasets.HuggingFaceDataset(hugDataset)
