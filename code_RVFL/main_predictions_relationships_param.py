from data_set import TrainingDataSet, TestDataSet
import activation_function
import rvfl
import numpy as np
import matplotlib.pyplot as plt
import kFoldValidation
import pandas as pd
import math


def RVFL(num_of_hidden_nodes, seed, path_training, dataToBePredicted):

    norm_range_min, norm_range_max = 0, 1

    training_data = TrainingDataSet(
        path_training, norm_range_min, norm_range_max)

    # min and max from training data
    training_max_y = training_data.y_max
    training_min_y = training_data.y_min
    training_max_x = training_data.x_max
    training_min_x = training_data.x_min

    norm_range_diff = norm_range_max-norm_range_min

    x_train, y_train = training_data.x, training_data.y
    x_test = ((dataToBePredicted - training_min_x) /
              (training_max_x - training_min_x)) * norm_range_diff + norm_range_min

    model_name = "HL" + str(num_of_hidden_nodes) + "seed" + \
        str(seed) + "numOfData" + str(numOfTrainingData)

    RVFL_model = rvfl.RVFL(numberOfNodes=num_of_hidden_nodes, seed=seed,
                           activation_function=activation_function.relu)

    RVFL_model.train(x_train, y_train)
    predicted_output = RVFL_model.predict(x_test)

    output_nonNormalized = ((predicted_output - norm_range_min)/norm_range_diff) * \
        (training_max_y-training_min_y) + training_min_y

    return output_nonNormalized


num_of_hidden_nodes = 19
seed = 10

numOfTrainingData = 20
path_training = "data/dataWithDepthAndDiameter_TRAININGplusVALIDATON_new_1.xls"
#path_training = "datas_random_3/dataWithDepthAndDiameter_TRAININGplusVALIDATON_new_1_20.xls"
randomDataSet_num = 1
path_dataToBePredicted = "dataToBePredicted/differing_repetition_rate_1.xls"

# read the excel file with pandas
df = pd.read_excel(path_dataToBePredicted)
# convert it to numpy
dataToBePredicted = df.to_numpy()


# power, pulse repetition rate 1200 and 600
path_dataToBePredicted_5_prr_600 = "dataToBePredicted/differing_power_PRR_600_PD_5_DT_03.xls"
path_dataToBePredicted_015_prr_600 = "dataToBePredicted/differing_power_PRR_600_PD_015_DT_03.xls"
path_dataToBePredicted_5_prr_1200 = "dataToBePredicted/differing_power_PRR_1200_PD_5_DT_03.xls"
path_dataToBePredicted_015_prr_1200 = "dataToBePredicted/differing_power_PRR_1200_PD_015_DT_03.xls"
path_dataToBePredicted_2_prr_600 = "dataToBePredicted/differing_power_PRR_600_PD_2_DT_03.xls"
path_dataToBePredicted_2_prr_1200 = "dataToBePredicted/differing_power_PRR_1200_PD_2_DT_03.xls"
path_dataToBePredicted_1_prr_600 = "dataToBePredicted/differing_power_PRR_600_PD_1_DT_03.xls"
path_dataToBePredicted_1_prr_1200 = "dataToBePredicted/differing_power_PRR_1200_PD_1_DT_03.xls"

# read the excel file with pandas
df_5_prr_600 = pd.read_excel(path_dataToBePredicted_5_prr_600)
df_015_prr_600 = pd.read_excel(path_dataToBePredicted_015_prr_600)
df_5_prr_1200 = pd.read_excel(path_dataToBePredicted_5_prr_1200)
df_015_prr_1200 = pd.read_excel(path_dataToBePredicted_015_prr_1200)
df_2_prr_600 = pd.read_excel(path_dataToBePredicted_2_prr_600)
df_2_prr_1200 = pd.read_excel(path_dataToBePredicted_2_prr_1200)
df_1_prr_600 = pd.read_excel(path_dataToBePredicted_1_prr_600)
df_1_prr_1200 = pd.read_excel(path_dataToBePredicted_1_prr_1200)
# convert it to numpy
dataToBePredicted_5_prr_600 = df_5_prr_600.to_numpy()
dataToBePredicted_015_prr_600 = df_015_prr_600.to_numpy()
dataToBePredicted_5_prr_1200 = df_5_prr_1200.to_numpy()
dataToBePredicted_015_prr_1200 = df_015_prr_1200.to_numpy()
dataToBePredicted_2_prr_600 = df_2_prr_600.to_numpy()
dataToBePredicted_2_prr_1200 = df_2_prr_1200.to_numpy()
dataToBePredicted_1_prr_600 = df_1_prr_600.to_numpy()
dataToBePredicted_1_prr_1200 = df_1_prr_1200.to_numpy()


model_name = "HL" + str(num_of_hidden_nodes) + "seed" + \
    str(seed) + "numOfData" + str(numOfTrainingData)

predictions_5_prr_600 = RVFL(
    num_of_hidden_nodes, seed, path_training, dataToBePredicted_5_prr_600)
predictions_015_prr_600 = RVFL(
    num_of_hidden_nodes, seed, path_training, dataToBePredicted_015_prr_600)
predictions_2_prr_600 = RVFL(
    num_of_hidden_nodes, seed, path_training, dataToBePredicted_2_prr_600)
predictions_1_prr_600 = RVFL(
    num_of_hidden_nodes, seed, path_training, dataToBePredicted_1_prr_600)
predictions_5_prr_1200 = RVFL(
    num_of_hidden_nodes, seed, path_training, dataToBePredicted_5_prr_1200)
predictions_2_prr_1200 = RVFL(
    num_of_hidden_nodes, seed, path_training, dataToBePredicted_2_prr_1200)
predictions_015_prr_1200 = RVFL(
    num_of_hidden_nodes, seed, path_training, dataToBePredicted_015_prr_1200)
predictions_1_prr_1200 = RVFL(
    num_of_hidden_nodes, seed, path_training, dataToBePredicted_1_prr_1200)


t_drill = 0.3

plt.figure(7)
plt.xlabel('Energy [mJ]')
plt.ylabel('Diameter [µm]')
plt.plot(dataToBePredicted_015_prr_1200[:, 0]*t_drill, predictions_015_prr_1200[:, 0],
         'x', label='Pulse Duration = 0,15 ns')
plt.plot(dataToBePredicted_1_prr_1200[:, 0]*t_drill, predictions_1_prr_1200[:, 0],
         'x', label='Pulse Duration = 1 ns')
plt.plot(dataToBePredicted_2_prr_1200[:, 0]*t_drill, predictions_2_prr_1200[:, 0],
         'x', label='Pulse Duration = 2 ns')
plt.plot(dataToBePredicted_5_prr_1200[:, 0]*t_drill, predictions_5_prr_1200[:, 0],
         'x', label='Pulse Duration = 5 ns')
#plt.title(" Power vs Diameter for different PRRs")
plt.legend()
plt.show()


plt.figure(8)
plt.xlabel('Energy [mJ]')
plt.ylabel('Diameter [µm]')
plt.plot(dataToBePredicted_015_prr_600[:, 0]*t_drill, predictions_015_prr_600[:, 0],
         'x', label='Pulse Duration = 0,15 ns')
plt.plot(dataToBePredicted_1_prr_600[:, 0]*t_drill, predictions_1_prr_600[:, 0],
         'x', label='Pulse Duration = 1 ns')
plt.plot(dataToBePredicted_2_prr_600[:, 0]*t_drill, predictions_2_prr_600[:, 0],
         'x', label='Pulse Duration = 2 ns')
plt.plot(dataToBePredicted_5_prr_600[:, 0]*t_drill, predictions_5_prr_600[:, 0],
         'x', label='Pulse Duration = 5 ns')
#plt.title(" Power vs Diameter for different PRRs")
plt.legend()
plt.show()

print("0.15: ", predictions_015_prr_600[:, 0])
print("1: ", predictions_1_prr_600[:, 0])
print("2: ", predictions_2_prr_600[:, 0])
print("5: ", predictions_5_prr_600[:, 0])
# power
path_dataToBePredicted_5 = "dataToBePredicted/differing_power_1.xls"
path_dataToBePredicted_015 = "dataToBePredicted/differing_power_2.xls"

# read the excel file with pandas
df_5 = pd.read_excel(path_dataToBePredicted_5)
df_015 = pd.read_excel(path_dataToBePredicted_015)

# convert it to numpy
dataToBePredicted_5 = df_5.to_numpy()
dataToBePredicted_015 = df_015.to_numpy()


fluenz_5 = 2 * dataToBePredicted_5[:,
                                   0] / ((((27.1/2) * (10**-4))**2)*math.pi*1200*(10**3))
fluenz_015 = 2 * dataToBePredicted_015[:,
                                       0] / ((((27.1/2) * (10**-4))**2)*math.pi*1200*(10**3))
predictions_5 = RVFL(
    num_of_hidden_nodes, seed, path_training, dataToBePredicted_5)
predictions_015 = RVFL(
    num_of_hidden_nodes, seed, path_training, dataToBePredicted_015)
