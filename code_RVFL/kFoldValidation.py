import numpy as np


def kFoldDataSplit(data, fold_idx, numOfFold, input_size, norm_range_min, norm_range_max):

    norm_range_diff = norm_range_max - norm_range_min
    for i in range(data.shape[1]):
        data[:, i] = ((data[:, i] - min(data[:, i])) *
                      (1.0 / (max(data[:, i]) - min(data[:, i])))) * norm_range_diff + norm_range_min

    idx_full = np.arange(len(data))

    np.random.seed(0)
    np.random.shuffle(idx_full)  # shuffle the array of indexes

    fold_length = int(len(data) / numOfFold)

    valid_idx = idx_full[fold_length * fold_idx: fold_length * (fold_idx + 1)]
    train_idx = np.delete(idx_full, np.arange(
        fold_length * fold_idx, fold_length * (fold_idx + 1)))

    # validation input and output
    x_valid, y_valid = (data[valid_idx, :input_size],
                        data[valid_idx, input_size:])
    # train input and output
    x_train, y_train = (data[train_idx, :input_size],
                        data[train_idx, input_size:])

    print("fold idx ", fold_idx)

    return x_valid, y_valid, x_train, y_train
