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
    print("model name: ", model_name)

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

    ae_diameter = abs(
        target_nonNormalized[:, 0] - output_nonNormalized[:, 0])
    ae_depth = abs(
        target_nonNormalized[:, 1] - output_nonNormalized[:, 1])

    return ae_diameter, ae_depth


num_of_hidden_nodes = 19
seed = 10

numOfTrainingData = 20
path_training = "data/dataWithDepthAndDiameter_TRAININGplusVALIDATON_new_1.xls"
randomDataSet_num = 1
path_test = "data/dataWithDepthAndDiameter_TEST_new_1.xls"
model_name = "HL" + str(num_of_hidden_nodes) + "seed" + \
    str(seed) + "numOfData" + str(numOfTrainingData)
# k fold cross validation
kfold = False
ae_predictedRVFL_20 = RVFL(num_of_hidden_nodes, seed, "datas_random_"+str(randomDataSet_num) +
                           "/dataWithDepthAndDiameter_TRAININGplusVALIDATON_new_1_20.xls", path_test, kfold)
ae_predictedRVFL_30 = RVFL(num_of_hidden_nodes, seed, "datas_random_"+str(randomDataSet_num) +
                           "/dataWithDepthAndDiameter_TRAININGplusVALIDATON_new_1_30.xls", path_test, kfold)
ae_predictedRVFL_40 = RVFL(num_of_hidden_nodes, seed, "datas_random_"+str(randomDataSet_num) +
                           "/dataWithDepthAndDiameter_TRAININGplusVALIDATON_new_1_40.xls", path_test, kfold)
ae_predictedRVFL_50 = RVFL(num_of_hidden_nodes, seed, "datas_random_"+str(randomDataSet_num) +
                           "/dataWithDepthAndDiameter_TRAININGplusVALIDATON_new_1_50.xls", path_test, kfold)
ae_predictedRVFL_60 = RVFL(num_of_hidden_nodes, seed, "datas_random_"+str(randomDataSet_num) +
                           "/dataWithDepthAndDiameter_TRAININGplusVALIDATON_new_1_60.xls", path_test, kfold)
ae_predictedRVFL_full = RVFL(
    num_of_hidden_nodes, seed, path_training, path_test, kfold)

ae_diameter_predictedRVFL_20, ae_depth_predictedRVFL_20 = ae_predictedRVFL_20[
    0], ae_predictedRVFL_20[1]
ae_diameter_predictedRVFL_30, ae_depth_predictedRVFL_30 = ae_predictedRVFL_30[
    0], ae_predictedRVFL_30[1]
ae_diameter_predictedRVFL_40, ae_depth_predictedRVFL_40 = ae_predictedRVFL_40[
    0], ae_predictedRVFL_40[1]
ae_diameter_predictedRVFL_50, ae_depth_predictedRVFL_50 = ae_predictedRVFL_50[
    0], ae_predictedRVFL_50[1]
ae_diameter_predictedRVFL_60, ae_depth_predictedRVFL_60 = ae_predictedRVFL_60[
    0], ae_predictedRVFL_60[1]
ae_diameter_predictedRVFL_full, ae_depth_predictedRVFL_full = ae_predictedRVFL_full[
    0], ae_predictedRVFL_full[1]

x_pos = np.array([20, 30, 40, 50, 60, 73])


args_diameter = (ae_diameter_predictedRVFL_20, ae_diameter_predictedRVFL_30,
                 ae_diameter_predictedRVFL_40, ae_diameter_predictedRVFL_50,
                 ae_diameter_predictedRVFL_60, ae_diameter_predictedRVFL_full)

data_diameter = np.concatenate(args_diameter)

args_depth = (ae_depth_predictedRVFL_20, ae_depth_predictedRVFL_30, ae_depth_predictedRVFL_40,
              ae_depth_predictedRVFL_50, ae_depth_predictedRVFL_60, ae_depth_predictedRVFL_full)

data_depth = np.concatenate(args_depth)

mlmodels = []
j = 0
for i in range((len(x_pos)*len(ae_diameter_predictedRVFL_20))):

    if(i % 8 == 0 and i > 6):
        j = j+1
    mlmodels.append(x_pos[j])

df = pd.DataFrame(
    {'Number of Training Data': mlmodels, 'Diameter': data_diameter, 'Depth': data_depth})

dd = pd.melt(df, id_vars=['Number of Training Data'], value_vars=[
             'Diameter', 'Depth'], var_name='Predicted values')
sns.boxplot(x='Number of Training Data', y='value',
            data=dd, hue='Predicted values')

plt.xlabel("Number of Training Data", fontsize=13)
plt.ylabel("Absolute Error in µm", fontsize=13)
plt.tick_params(axis='both', labelsize=13)
plt.legend(fontsize=13)
plt.ylim([-1, 60])
plt.savefig("boxPlot_NNModel_RVFL_differentNumOfData_rand_" + str(randomDataSet_num) + ".png", bbox_inches='tight',
            pad_inches=0.1)
plt.show()
