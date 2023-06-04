"""A Dataset that returns a batch of data with their indices"""

from torch.utils.data import Dataset


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.targets = dataset.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index
