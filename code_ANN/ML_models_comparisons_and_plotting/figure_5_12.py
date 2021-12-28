
import NNmodelsPredictedRelationships
import pandas as pd
import matplotlib.pyplot as plt
import math
# pulse repetition rate 1200
path_dataToBePredicted_5 = "dataToBePredicted/differing_repetition_rate_1.xls"
path_dataToBePredicted_015 = "dataToBePredicted/differing_repetition_rate_2.xls"

# read the excel file with pandas
df_5 = pd.read_excel(path_dataToBePredicted_5)
df_015 = pd.read_excel(path_dataToBePredicted_015)

# convert it to numpy
dataToBePredicted_5 = df_5.to_numpy()
dataToBePredicted_015 = df_015.to_numpy()


model_name = "383_new_full_20_3"
predictions_5 = NNmodelsPredictedRelationships.predict(
    model_name, dataToBePredicted_5)
predictions_015 = NNmodelsPredictedRelationships.predict(
    model_name, dataToBePredicted_015)


# power
path_dataToBePredicted_5 = "dataToBePredicted/differing_power_1.xls"
path_dataToBePredicted_015 = "dataToBePredicted/differing_power_2.xls"

# read the excel file with pandas
df_5 = pd.read_excel(path_dataToBePredicted_5)
df_015 = pd.read_excel(path_dataToBePredicted_015)

# convert it to numpy
dataToBePredicted_5 = df_5.to_numpy()
dataToBePredicted_015 = df_015.to_numpy()


predictions_5 = NNmodelsPredictedRelationships.predict(
    model_name, dataToBePredicted_5)
predictions_015 = NNmodelsPredictedRelationships.predict(
    model_name, dataToBePredicted_015)


fluenz_5 = 2 * dataToBePredicted_5[:,
                                   0] / ((((27.1/2) * (10**-4))**2)*math.pi*1200*(10**3))
fluenz_015 = 2 * dataToBePredicted_015[:,
                                       0] / ((((27.1/2) * (10**-4))**2)*math.pi*1200*(10**3))


# power, pulse repetition rate 1200 and 600
path_dataToBePredicted_5_prr_600 = "dataToBePredicted/differing_power_3.xls"
path_dataToBePredicted_015_prr_600 = "dataToBePredicted/differing_power_4.xls"
path_dataToBePredicted_5_prr_1200 = "dataToBePredicted/differing_power_5.xls"
path_dataToBePredicted_015_prr_1200 = "dataToBePredicted/differing_power_6.xls"
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

model_name = "383_new_full"
predictions_5_prr_600 = NNmodelsPredictedRelationships.predict(
    model_name, dataToBePredicted_5_prr_600)
predictions_015_prr_600 = NNmodelsPredictedRelationships.predict(
    model_name, dataToBePredicted_015_prr_600)
predictions_5_prr_1200 = NNmodelsPredictedRelationships.predict(
    model_name, dataToBePredicted_5_prr_1200)
predictions_015_prr_1200 = NNmodelsPredictedRelationships.predict(
    model_name, dataToBePredicted_015_prr_1200)
predictions_2_prr_600 = NNmodelsPredictedRelationships.predict(
    model_name, dataToBePredicted_2_prr_600)
predictions_2_prr_1200 = NNmodelsPredictedRelationships.predict(
    model_name, dataToBePredicted_2_prr_1200)
predictions_1_prr_600 = NNmodelsPredictedRelationships.predict(
    model_name, dataToBePredicted_1_prr_600)
predictions_1_prr_1200 = NNmodelsPredictedRelationships.predict(
    model_name, dataToBePredicted_1_prr_1200)

t_drill = 0.3

RVFL_predictions_015_prr_600 = [61.39652467,  80.66885371,  92.66043997, 104.65202624, 114.24529525,
                                128.63519877,  55.48758383,  43.66970217,  31.85182051, 121.44024701,
                                76.86035022,  99.85539173, 107.05034349]
RVFL_predictions_1_prr_600 = [60.49645515,  81.04967052, 102.03058728, 114.02217354, 123.61544256,
                              138.00534608,  54.64373774,  42.82585608,  31.00797441, 130.81039432,
                              75.91136668, 109.22553904, 116.4204908]
RVFL_predictions_2_prr_600 = [59.38000393,  79.9332193,  105.62473852, 125.04587625, 134.63914526,
                              150.0185003,   53.65097763,  41.83309597,  30.0152143,  141.83409702,
                              74.79491546, 120.24924175, 127.4441935]
RVFL_predictions_5_prr_600 = [57.23078811,  79.13151207, 107.10677382, 135.08203558, 157.46224499,
                              191.03255909,  51.77496437,  40.4109343,   29.31080811, 174.24740204,
                              73.53645972, 123.89193088, 140.67708793]


experiment_015 = [110, 106, 104,  83,  65]
experiment_1 = [124, 118, 101,  86,  61]
experiment_2 = [131, 130, 120, 103,  72]
experiment_5 = [146, 118, 104,  89,  72]

energy = [9,  7.2, 5.4, 3.6, 1.8]

plt.figure(1)
plt.plot(dataToBePredicted_015_prr_600[:, 0]*t_drill, predictions_015_prr_600[:, 0],
         's', label='ANN Model')
plt.plot(dataToBePredicted_015_prr_600[:, 0]*t_drill, RVFL_predictions_015_prr_600,
         's', label='RVFL Model')
plt.plot(energy, experiment_015,
         's', label='Experiments')
plt.ylabel('Diameter [µm]', fontsize=13)
plt.xlabel('Energy [mJ]', fontsize=13)
plt.tick_params(axis='both', labelsize=13)
plt.legend(fontsize=13)
plt.show()


plt.figure(2)
plt.plot(dataToBePredicted_1_prr_600[:, 0]*t_drill, predictions_1_prr_600[:, 0],
         's', label='ANN Model')
plt.plot(dataToBePredicted_1_prr_600[:, 0]*t_drill, RVFL_predictions_1_prr_600,
         's', label='RVFL Model')
plt.plot(energy, experiment_1,
         's', label='Experiments')
plt.ylabel('Diameter [µm]', fontsize=13)
plt.xlabel('Energy [mJ]', fontsize=13)
plt.tick_params(axis='both', labelsize=13)
plt.legend(fontsize=13)
plt.show()

plt.figure(3)
plt.plot(dataToBePredicted_2_prr_600[:, 0]*t_drill, predictions_2_prr_600[:, 0],
         's', label='ANN Model')
plt.plot(dataToBePredicted_2_prr_600[:, 0]*t_drill, RVFL_predictions_2_prr_600,
         's', label='RVFL Model')
plt.plot(energy, experiment_2,
         's', label='Experiments')
plt.ylabel('Diameter [µm]', fontsize=13)
plt.xlabel('Energy [mJ]', fontsize=13)
plt.tick_params(axis='both', labelsize=13)
plt.legend(fontsize=13)
plt.show()

plt.figure(4)
plt.plot(dataToBePredicted_5_prr_600[:, 0]*t_drill, predictions_5_prr_600[:, 0],
         's', label='ANN Model')
plt.plot(dataToBePredicted_5_prr_600[:, 0]*t_drill, RVFL_predictions_5_prr_600,
         's', label='RVFL Model')
plt.plot(energy, experiment_5,
         's', label='Experiments')
plt.ylabel('Diameter [µm]', fontsize=13)
plt.xlabel('Energy [mJ]', fontsize=13)
plt.tick_params(axis='both', labelsize=13)
plt.legend(fontsize=13)
plt.show()
