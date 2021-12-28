from data_set import TrainingDataSet, TestDataSet
import activation_function
import rvfl
import numpy as np
import matplotlib.pyplot as plt
import kFoldValidation
import pandas as pd
import seaborn as sns


def RVFL(num_of_hidden_nodes, seed, path_training, path_test, kfold):

    norm_range_min, norm_range_max = 0, 1

    training_data = TrainingDataSet(
        path_training, norm_range_min, norm_range_max)
    test_data = TestDataSet(path_test, norm_range_min,
                            norm_range_max, training_data)

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

    model_name = "HL" + str(num_of_hidden_nodes) + "seed" + \
        str(seed) + "numOfData" + str(numOfTrainingData)
    # print("model name: ", model_name)

    RVFL_model = rvfl.RVFL(numberOfNodes=num_of_hidden_nodes, seed=seed,
                           activation_function=activation_function.relu)

    RVFL_model.train(x_train, y_train)
    predicted_output = RVFL_model.predict(x_test)
    mse, mae = RVFL_model.eval(predicted_output, y_test)

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

    return output_nonNormalized, abs(output_nonNormalized - target_nonNormalized)


num_of_hidden_nodes = 19
seed = 10

numOfTrainingData = 20
randomDataSet_num = 4


# k fold cross validation
kfold = False
ae_diam = []
ae_depth = []


for randomDataSet_num in range(1, 5):

    path_training = "20_data_point_ananalysis/20_data_points_set_" + str(randomDataSet_num) + \
        "/dataWithDepthAndDiameter_TRAININGplusVALIDATON_new_Analysis_20.xls"
    path_test = "20_data_point_ananalysis/20_data_points_set_" + str(randomDataSet_num) + \
        "/dataWithDepthAndDiameter_TEST_new_Analysis_20.xls"

    ae_predictions = RVFL(num_of_hidden_nodes, seed,
                          path_training, path_test, kfold)[1]
    ae_diam.append(ae_predictions[:, 0])

    ae_depth.append(ae_predictions[:, 1])


x_pos = np.array([1, 2, 3, 4])

ae_diam = np.concatenate(ae_diam)
ae_depth = np.concatenate(ae_depth)


sets = []
j = 0
for i in range((len(x_pos)*61)):

    if(i % 61 == 0 and i > 59):
        j = j+1
    sets.append(x_pos[j])


df = pd.DataFrame(
    {'Random Sets': sets, 'Diameter': ae_diam, 'Depth': ae_depth})

dd = pd.melt(df, id_vars=['Random Sets'], value_vars=[
             'Diameter', 'Depth'], var_name='Predicted values')
sns.boxplot(x='Random Sets', y='value',
            data=dd, hue='Predicted values')

plt.ylim([0, 60])
plt.ylabel("Absolute Error in µm")
plt.savefig("boxPlot_RVFL_differentNumOfData_train_20_test_61.png", bbox_inches='tight',
            pad_inches=0.1)
plt.show()
