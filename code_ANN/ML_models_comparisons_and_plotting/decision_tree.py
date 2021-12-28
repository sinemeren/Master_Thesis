
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

data_dir = "data/"
file_name_train = 'dataWithDepthAndDiameter_TRAININGplusVALIDATON_new_1.xls'
input_size = 4

# read the excel file with pandas
df_train = pd.read_excel(data_dir + file_name_train)

# convert it to numpy
data_train = df_train.to_numpy()

# input and output
x_train, y_train = (data_train[:, :input_size], data_train[:, input_size:])


model = tree.DecisionTreeRegressor(max_depth=6)
model.fit(x_train, y_train)


file_name_test = 'dataWithDepthAndDiameter_TEST_new_1.xls'

# read the excel file with pandas
df_test = pd.read_excel(data_dir + file_name_test)

# convert it to numpy
data_test = df_test.to_numpy()

# input and output
x_test, y_test = (data_test[:, :input_size], data_test[:, input_size:])

y_predicted = model.predict(x_test)

print("depth is", model.get_depth())
print("target", y_test)
print("predictions", y_predicted)

absolute_error = abs(y_predicted - y_test)
print("absolute error", absolute_error)

print("average prediction error", np.mean(absolute_error, axis=0))

x_pos = np.linspace(0, len(y_test)-1, len(y_test))

plt.xlabel('Testing Index')
plt.ylabel('Diameter')
plt.plot(x_pos, y_predicted[:, 0], 'x', label='Predicted Data')
plt.plot(x_pos, y_test[:, 0], '^', label='Experimental Data')
plt.title("Decision Tree Regression - Diameter")
plt.legend()
plt.show()

plt.xlabel('Testing Index')
plt.ylabel('Depth')
plt.plot(x_pos, y_predicted[:, 1], 'x', label='Predicted Data')
plt.plot(x_pos, y_test[:, 1], '^', label='Experimental Data')
plt.title("Decision Tree Regression - Depth")
plt.legend()
plt.show()


def decision_tree_kfold(data, fold_idx, numOfFold, depth_max):

    idx_full = np.arange(len(data))
    input_size = 4

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

    model = tree.DecisionTreeRegressor(max_depth=depth_max)
    model.fit(x_train, y_train)

    y_predicted = model.predict(x_valid)

    print("fold idx ", fold_idx)
    print("average prediction error ", mean_absolute_error(y_valid, y_predicted))


depth_max = 10
decision_tree_kfold(data_train, fold_idx=0, numOfFold=5, depth_max=depth_max)
decision_tree_kfold(data_train, fold_idx=1, numOfFold=5, depth_max=depth_max)
decision_tree_kfold(data_train, fold_idx=2, numOfFold=5, depth_max=depth_max)
decision_tree_kfold(data_train, fold_idx=3, numOfFold=5, depth_max=depth_max)
decision_tree_kfold(data_train, fold_idx=4, numOfFold=5, depth_max=depth_max)
