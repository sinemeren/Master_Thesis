from data_set import TrainingDataSet, TestDataSet
import activation_function
import rvfl
import numpy as np
import matplotlib.pyplot as plt
import kFoldValidation
import pandas as pd
from sklearn.metrics import r2_score

num_of_hidden_nodes = 19

path_training = "data/dataWithDepthAndDiameter_TRAININGplusVALIDATON_new_1.xls"


path_test = "data/dataWithDepthAndDiameter_TEST_new_1.xls"
norm_range_min, norm_range_max = 0, 1

training_data = TrainingDataSet(path_training, norm_range_min, norm_range_max)
test_data = TestDataSet(path_test, norm_range_min,
                        norm_range_max, training_data)

# k fold cross validation
kfold = False

x_train, y_train = training_data.x, training_data.y
x_test, y_test = test_data.x, test_data.y

if(kfold):

    # read the excel file with pandas
    df = pd.read_excel(path_training)

    # convert it to numpy
    data = df.to_numpy()

    fold_idx = 4
    numOfFold = 5
    input_size = 4
    x_valid, y_valid, x_train, y_train = kFoldValidation.kFoldDataSplit(
        data, fold_idx, numOfFold, input_size, norm_range_min, norm_range_max)
    x_test = x_valid
    y_test = y_valid


seed = 10

RVFL_model = rvfl.RVFL(numberOfNodes=num_of_hidden_nodes, seed=seed,
                       activation_function=activation_function.relu)

RVFL_model.train(x_train, y_train)
predicted_output = RVFL_model.predict(x_test)
mse, mae = RVFL_model.eval(predicted_output, y_test)

print("mse: " + str(mse) + " , mae: " + str(mae))

norm_range_diff = norm_range_max-norm_range_min
# min and max from training data
training_max_x = training_data.x_max
training_max_y = training_data.y_max
training_min_x = training_data.x_min
training_min_y = training_data.y_min

output_nonNormalized = ((predicted_output - norm_range_min)/norm_range_diff) * \
    (training_max_y-training_min_y) + training_min_y
target_nonNormalized = ((y_test - norm_range_min)/norm_range_diff) * \
    (training_max_y-training_min_y) + training_min_y

print("predictions" + str(output_nonNormalized))
print("test: " + str(target_nonNormalized))

x_pos = np.linspace(0, len(y_test)-1, len(y_test))

plt.figure(1)
plt.xlabel('Testing Index')
plt.ylabel('Diameter [µm]')
plt.plot(x_pos, output_nonNormalized[:, 0], 'x', label='Predicted Data')
plt.plot(x_pos, target_nonNormalized[:, 0], '^', label='Experimental Data')
plt.legend()

plt.figure(2)
plt.xlabel('Testing Index')
plt.ylabel('Depth [µm]')
plt.plot(x_pos, output_nonNormalized[:, 1], 'x', label='Predicted Data')
plt.plot(x_pos, target_nonNormalized[:, 1], '^', label='Experimental Data')
plt.legend()
plt.show()

# r2
r_squared = r2_score(target_nonNormalized, output_nonNormalized)
plt.figure(3)

plt.scatter(target_nonNormalized[:, 0], output_nonNormalized[:, 0],
            marker='o', label="Diameter")
plt.scatter(target_nonNormalized[:, 1], output_nonNormalized[:, 1],
            marker='s', label="Depth")

plt.ylabel("Predicted diameters")
plt.xlabel("Actual diameters")
plt.text(90, 50, 'R-squared = %0.2f' % r_squared)
plt.plot((np.array((0, 120))), (np.array((0, 120))), '--', color='k')
plt.legend()
plt.show()
