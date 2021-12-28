from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np


class ProcessParamDataSet(Dataset):
    def __init__(self, data_dir, norm_range_min, norm_range_max, file_name) -> None:

        #file_name = 'dataWithDepthAndDiameter_TRAININGplusVALIDATON_new_1.xls'
        input_size = 4

        # read the excel file with pandas
        self.df = pd.read_excel(data_dir + file_name)

        # convert it to numpy
        self.data = self.df.to_numpy()

        # max and min of dataset
        self.x_max = self.data[:, :input_size].max(0)
        self.x_min = self.data[:, :input_size].min(0)

        self.y_max = self.data[:, input_size:].max(0)
        self.y_min = self.data[:, input_size:].min(0)

        # convert it to torch tensor and divide as input and output
        self.x, self.y = (torch.from_numpy(
            self.data[:, :input_size]), torch.from_numpy(self.data[:, input_size:]))

        # conver the dataset to float
        self.x = self.x.float()
        self.y = self.y.float()

        # normalization range
        self.norm_range_min = norm_range_min
        self.norm_range_max = norm_range_max
        self.norm_range_diff = self.norm_range_max - self.norm_range_min

        # normalization
        for i in range(self.x.shape[1]):
            self.x[:, i] = ((self.x[:, i] - min(self.x[:, i])) *
                            (1.0 / (max(self.x[:, i]) - min(self.x[:, i])))) * self.norm_range_diff + self.norm_range_min

        for i in range(self.y.shape[1]):
            self.y[:, i] = ((self.y[:, i] - min(self.y[:, i])) *
                            (1.0 / (max(self.y[:, i]) - min(self.y[:, i])))) * self.norm_range_diff + self.norm_range_min

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return len(self.data)


class ProcessParamDataSetTEST_20_Analysis(Dataset):
    def __init__(self, data_dir, norm_range_min, norm_range_max, trainingDataSet) -> None:

        file_name = 'dataWithDepthAndDiameter_TEST_new_Analysis_20.xls'

        print("data directory: ", data_dir)

        input_size = 4

        # read the excel file with pandas
        self.df = pd.read_excel(data_dir + file_name)

        # convert it to numpy
        self.data = self.df.to_numpy()
        print("Length of data ", len(self.data))
        # convert it to torch tensor and divide as input and output
        self.x, self.y = (torch.from_numpy(
            self.data[:, :input_size]), torch.from_numpy(self.data[:, input_size:]))

        # conver the dataset to float
        self.x = self.x.float()
        self.y = self.y.float()

        # normalization range
        self.norm_range_min = norm_range_min
        self.norm_range_max = norm_range_max
        self.norm_range_diff = self.norm_range_max - self.norm_range_min

        # normalization
        # min and max from training data
        training_max_x = trainingDataSet.x_max
        training_max_y = trainingDataSet.y_max
        training_min_x = trainingDataSet.x_min
        training_min_y = trainingDataSet.y_min

        training_min_x = torch.from_numpy(np.asarray(training_min_x))
        training_max_x = torch.from_numpy(np.asarray(training_max_x))
        training_min_y = torch.from_numpy(np.asarray(training_min_y))
        training_max_y = torch.from_numpy(np.asarray(training_max_y))

        for i in range(self.x.shape[1]):
            self.x[:, i] = ((self.x[:, i] - training_min_x[i]) *
                            (1.0 / (training_max_x[i] - training_min_x[i]))) * self.norm_range_diff + self.norm_range_min

        for i in range(self.y.shape[1]):
            self.y[:, i] = ((self.y[:, i] - training_min_y[i]) *
                            (1.0 / (training_max_y[i] - training_min_y[i]))) * self.norm_range_diff + self.norm_range_min

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return len(self.data)


class ProcessParamDataSetTEST(Dataset):
    def __init__(self, data_dir, norm_range_min, norm_range_max, trainingDataSet) -> None:

        file_name = 'dataWithDepthAndDiameter_TEST_new_1.xls'

        input_size = 4

        # read the excel file with pandas
        self.df = pd.read_excel(data_dir + file_name)

        # convert it to numpy
        self.data = self.df.to_numpy()

        # convert it to torch tensor and divide as input and output
        self.x, self.y = (torch.from_numpy(
            self.data[:, :input_size]), torch.from_numpy(self.data[:, input_size:]))

        # conver the dataset to float
        self.x = self.x.float()
        self.y = self.y.float()

        # normalization range
        self.norm_range_min = norm_range_min
        self.norm_range_max = norm_range_max
        self.norm_range_diff = self.norm_range_max - self.norm_range_min

        # normalization
        # min and max from training data
        training_max_x = trainingDataSet.x_max
        training_max_y = trainingDataSet.y_max
        training_min_x = trainingDataSet.x_min
        training_min_y = trainingDataSet.y_min

        training_min_x = torch.from_numpy(np.asarray(training_min_x))
        training_max_x = torch.from_numpy(np.asarray(training_max_x))
        training_min_y = torch.from_numpy(np.asarray(training_min_y))
        training_max_y = torch.from_numpy(np.asarray(training_max_y))

        for i in range(self.x.shape[1]):
            self.x[:, i] = ((self.x[:, i] - training_min_x[i]) *
                            (1.0 / (training_max_x[i] - training_min_x[i]))) * self.norm_range_diff + self.norm_range_min

        for i in range(self.y.shape[1]):
            self.y[:, i] = ((self.y[:, i] - training_min_y[i]) *
                            (1.0 / (training_max_y[i] - training_min_y[i]))) * self.norm_range_diff + self.norm_range_min

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return len(self.data)


class ProcessParamDataSetTEST_2(Dataset):
    def __init__(self, data_dir, norm_range_min, norm_range_max, trainingDataSet) -> None:

        # TODO: later take these from contructor
        file_name = 'dataWithDiameterOnly_diverse.xls'
        input_size = 4

        # read the excel file with pandas
        self.df = pd.read_excel(data_dir + file_name)

        # convert it to numpy
        self.data = self.df.to_numpy()

        # convert it to torch tensor and divide as input and output
        self.x, self.y = (torch.from_numpy(
            self.data[:, :input_size]), torch.from_numpy(self.data[:, input_size:]))

        # conver the dataset to float
        self.x = self.x.float()
        self.y = self.y.float()

        # normalization range
        self.norm_range_min = norm_range_min
        self.norm_range_max = norm_range_max
        self.norm_range_diff = self.norm_range_max - self.norm_range_min

        # normalization
        # min and max from training data
        training_max_x = trainingDataSet.x_max
        training_max_y = trainingDataSet.y_max
        training_min_x = trainingDataSet.x_min
        training_min_y = trainingDataSet.y_min

        training_min_x = torch.from_numpy(np.asarray(training_min_x))
        training_max_x = torch.from_numpy(np.asarray(training_max_x))
        training_min_y = torch.from_numpy(np.asarray(training_min_y))
        training_max_y = torch.from_numpy(np.asarray(training_max_y))

        for i in range(self.x.shape[1]):
            self.x[:, i] = ((self.x[:, i] - training_min_x[i]) *
                            (1.0 / (training_max_x[i] - training_min_x[i]))) * self.norm_range_diff + self.norm_range_min

        for i in range(self.y.shape[1]):
            self.y[:, i] = ((self.y[:, i] - training_min_y[i]) *
                            (1.0 / (training_max_y[i] - training_min_y[i]))) * self.norm_range_diff + self.norm_range_min

        print(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return len(self.data)
