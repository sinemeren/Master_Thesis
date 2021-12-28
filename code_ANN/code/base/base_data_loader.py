import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, fold_idx, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(
            self.validation_split, fold_idx)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        # creates train data loader
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split, fold_idx):
        if split == 0.0:
            return None, None

        # return array of indexes until the number of samples
        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)  # shuffle the array of indexes

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)  # length of validation

        '''
        # obtain the two array for indices, one for train and other for validation
        # takes samples from the shuffled array and creates a new array for the validation
        valid_idx = idx_full[0:len_valid]
        # delete the part that is allocated for the validation so the rest is for training
        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        '''

        valid_idx = idx_full[len_valid * fold_idx: len_valid * (fold_idx + 1)]
        train_idx = np.delete(idx_full, np.arange(
            len_valid * fold_idx, len_valid * (fold_idx + 1)))

        # creates samplers for train and validation
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False  # shuffle should be false since we use subset random sampler
        self.n_samples = len(train_idx)

        # return the samplers for train and validation
        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            # created validation data loader
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
