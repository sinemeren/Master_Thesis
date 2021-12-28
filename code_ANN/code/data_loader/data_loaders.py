from base import BaseDataLoader
from data_sets import ProcessParamDataSet, ProcessParamDataSetTEST, ProcessParamDataSetTEST_2, ProcessParamDataSetTEST_new_Nov_diff


class ProcessParamDataLoader(BaseDataLoader):
    def __init__(self, data_dir, file_name, fold_idx, norm_range_min, norm_range_max, batch_size=None, shuffle=True, k_fold=0,  validation_split=0.0, num_workers=1, training=True):

        self.data_dir = data_dir

        self.dataset = ProcessParamDataSet(
            data_dir, norm_range_min, norm_range_max, file_name)

        if(batch_size == None) or (batch_size == 0):
            batch_size = len(self.dataset)

        if (k_fold > 0):
            validation_split = 1 / k_fold
        else:
            fold_idx = 0

        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers, fold_idx)


class ProcessParamDataLoaderTEST_new_Nov_diff(BaseDataLoader):
    def __init__(self, data_dir, norm_range_min, norm_range_max, trainingDataSet, batch_size=None, shuffle=True,  validation_split=0.0, num_workers=1, training=True):

        self.data_dir = data_dir

        self.dataset = ProcessParamDataSetTEST_new_Nov_diff(
            data_dir, norm_range_min, norm_range_max, trainingDataSet)

        if(batch_size == None) or (batch_size == 0):
            batch_size = len(self.dataset)

        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers, 0)


class ProcessParamDataLoaderTEST(BaseDataLoader):
    def __init__(self, data_dir, norm_range_min, norm_range_max, trainingDataSet, batch_size=None, shuffle=True,  validation_split=0.0, num_workers=1, training=True):

        self.data_dir = data_dir

        self.dataset = ProcessParamDataSetTEST(
            data_dir, norm_range_min, norm_range_max, trainingDataSet)

        if(batch_size == None) or (batch_size == 0):
            batch_size = len(self.dataset)

        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers, 0)


class ProcessParamDataLoaderTEST_2(BaseDataLoader):
    def __init__(self, data_dir, norm_range_min, norm_range_max, trainingDataSet, batch_size=None, shuffle=True,  validation_split=0.0, num_workers=1, training=True):

        self.data_dir = data_dir

        self.dataset = ProcessParamDataSetTEST_2(
            data_dir, norm_range_min, norm_range_max, trainingDataSet)

        if(batch_size == None) or (batch_size == 0):
            batch_size = len(self.dataset)

        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers, 0)
