
import pandas as pd
import NNmodels
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = "data/"
input_size = 4

# load the test data
file_name_test = 'dataWithDepthAndDiameter_TEST_new_1.xls'

# read the excel file with pandas
df_test = pd.read_excel(data_dir + file_name_test)

# convert it to numpy
data_test = df_test.to_numpy()

# input and output
x_test, y_test = (data_test[:, :input_size], data_test[:, input_size:])


y_predicted_374_new_full = NNmodels.predict("374_new_full")
y_predicted_375_new_full = NNmodels.predict("375_new_full")
y_predicted_376_new_full = NNmodels.predict("376_new_full")
y_predicted_378_new_full = NNmodels.predict("378_new_full")
y_predicted_379_new_full = NNmodels.predict("379_new_full")
y_predicted_383_new_full = NNmodels.predict("383_new_full")


# HL: 19, seed:10 full
y_predicted_rvfl = np.array([[67.77074683, 41.50598147], [44.26832232, 29.06933324], [117.66775221, 62.39726218], [58.74097228, 43.7977239], [
                             78.54598809, 41.18530487], [32.58012983, 22.36925911], [67.36255973, 42.01421088], [82.27646078, 45.356633]])


ae_diameter_predicted374, ae_depth_predicted374 = abs(
    y_test[:, 0] - y_predicted_374_new_full[:, 0]), abs(y_test[:, 1] - y_predicted_374_new_full[:, 1])
ae_diameter_predicted375, ae_depth_predicted375 = abs(
    y_test[:, 0] - y_predicted_375_new_full[:, 0]), abs(y_test[:, 1] - y_predicted_375_new_full[:, 1])
ae_diameter_predicted376, ae_depth_predicted376 = abs(
    y_test[:, 0] - y_predicted_376_new_full[:, 0]), abs(y_test[:, 1] - y_predicted_376_new_full[:, 1])
ae_diameter_predicted378, ae_depth_predicted378 = abs(
    y_test[:, 0] - y_predicted_378_new_full[:, 0]), abs(y_test[:, 1] - y_predicted_378_new_full[:, 1])
ae_diameter_predicted379, ae_depth_predicted379 = abs(
    y_test[:, 0] - y_predicted_379_new_full[:, 0]), abs(y_test[:, 1] - y_predicted_379_new_full[:, 1])
ae_diameter_predicted383, ae_depth_predicted383 = abs(
    y_test[:, 0] - y_predicted_383_new_full[:, 0]), abs(y_test[:, 1] - y_predicted_383_new_full[:, 1])
ae_diameter_RVFL, ae_depth_RVFL = abs(
    y_test[:, 0] - y_predicted_rvfl[:, 0]), abs(y_test[:, 1] - y_predicted_rvfl[:, 1])


models = ['ANN Model 374', 'ANN Model 375',
          'ANN Model 376', 'ANN Model 378', 'ANN Model 379', 'ANN Model 383', 'RVFL']


args_diameter = (ae_diameter_predicted374, ae_diameter_predicted375,
                 ae_diameter_predicted376, ae_diameter_predicted378,
                 ae_diameter_predicted379, ae_diameter_predicted383, ae_diameter_RVFL)

data_diameter = np.concatenate(args_diameter)

args_depth = (ae_depth_predicted374, ae_depth_predicted375, ae_depth_predicted376,
              ae_depth_predicted378, ae_depth_predicted379, ae_depth_predicted383, ae_depth_RVFL)

data_depth = np.concatenate(args_depth)

mlmodels = []
j = 0
for i in range((len(models)*len(ae_diameter_predicted374))):

    if(i % 8 == 0 and i > 6):
        j = j+1
    mlmodels.append(models[j])

df = pd.DataFrame(
    {'Models': mlmodels, 'Diameter': data_diameter, 'Depth': data_depth})

dd = pd.melt(df, id_vars=['Models'], value_vars=[
             'Diameter', 'Depth'], var_name='Predicted values')
s = sns.boxplot(x='Models', y='value', data=dd, hue='Predicted values')
plt.ylabel("Absolute Error in µm", fontsize=13)
plt.tick_params(axis='both', labelsize=12)
s.set_xlabel("Models", fontsize=13)
plt.legend(fontsize=13)
plt.ylim([-1, 25])
plt.show()
