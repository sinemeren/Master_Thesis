import pandas as pd
import numpy as np


class TrainingDataSet:

    def __init__(self, path, norm_range_min, norm_range_max) -> None:

        self.input_size = 4

        # read the excel file with pandas
        self.df = pd.read_excel(path)

        # convert it to numpy
        self.data = self.df.to_numpy()

        # max and min of dataset
        self.x_max = self.data[:, :self.input_size].max(0)
        self.x_min = self.data[:, :self.input_size].min(0)

        self.y_max = self.data[:, self.input_size:].max(0)
        self.y_min = self.data[:, self.input_size:].min(0)

        # convert it to torch tensor and divide as input and output
        self.x, self.y = (self.data[:, :self.input_size],
                          self.data[:, self.input_size:])

        norm_range_diff = norm_range_max - norm_range_min

        # normalization
        for i in range(self.x.shape[1]):
            self.x[:, i] = ((self.x[:, i] - min(self.x[:, i])) *
                            (1.0 / (max(self.x[:, i]) - min(self.x[:, i])))) * norm_range_diff + norm_range_min

        for i in range(self.y.shape[1]):
            self.y[:, i] = ((self.y[:, i] - min(self.y[:, i])) *
                            (1.0 / (max(self.y[:, i]) - min(self.y[:, i])))) * norm_range_diff + norm_range_min


class TestDataSet:

    def __init__(self, test_path, norm_range_min, norm_range_max, trainingDataSet) -> None:

        input_size = 4

        # read the excel file with pandas
        self.df = pd.read_excel(test_path)

        # convert it to numpy
        self.data = self.df.to_numpy()

        self.x, self.y = (
            self.data[:, :input_size], self.data[:, input_size:])

        # normalization
        # min and max from training data
        training_max_x = trainingDataSet.x_max
        training_max_y = trainingDataSet.y_max
        training_min_x = trainingDataSet.x_min
        training_min_y = trainingDataSet.y_min

        self.norm_range_max = norm_range_max
        self.norm_range_min = norm_range_min
        self.norm_range_diff = self.norm_range_max - self.norm_range_min

        for i in range(self.x.shape[1]):
            self.x[:, i] = ((self.x[:, i] - training_min_x[i]) *
                            (1.0 / (training_max_x[i] - training_min_x[i]))) * self.norm_range_diff + self.norm_range_min

        for i in range(self.y.shape[1]):
            self.y[:, i] = ((self.y[:, i] - training_min_y[i]) *
                            (1.0 / (training_max_y[i] - training_min_y[i]))) * self.norm_range_diff + self.norm_range_min
