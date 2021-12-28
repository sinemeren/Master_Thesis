
import pandas as pd
import NNmodels
import numpy as np
import matplotlib.pyplot as plt

data_dir = "data/"
input_size = 4

# test data
file_name_test = 'dataWithDepthAndDiameter_TEST_new_1.xls'

# read the excel file with pandas
df_test = pd.read_excel(data_dir + file_name_test)

# convert it to numpy
data_test = df_test.to_numpy()

# input and output
x_test, y_test = (data_test[:, :input_size], data_test[:, input_size:])

# predictions from the NN model 383 trained with all data
y_predicted_383_new_full = NNmodels.predict("383_new_full")

# predictions from the RVFL model with Nodes: 19, seed:10 trained with all data
y_predicted_rvfl = np.array([[67.77074683, 41.50598147], [44.26832232, 29.06933324], [117.66775221, 62.39726218], [58.74097228, 43.7977239], [
                             78.54598809, 41.18530487], [32.58012983, 22.36925911], [67.36255973, 42.01421088], [82.27646078, 45.356633]])
x_pos = np.arange(len(x_test))+1


width = 0.25
fig, ax = plt.subplots()
ax.bar(x_pos-width, y_test[:, 0], width,
       capsize=5, label='Experimental Data', hatch='.')
ax.bar(x_pos,  y_predicted_383_new_full[:, 0], width,
       capsize=5, label='ANN Model predictions')
ax.bar(x_pos+width,  y_predicted_rvfl[:, 0], width,
       capsize=10, label='RVFL Model predictions')
ax.set_ylabel('Diameter [µm]', fontsize=13)
ax.set_xlabel('Testing Index', fontsize=13)
ax.legend(fontsize=11)
ax.tick_params(axis='both', labelsize=13)
plt.savefig("diameter_comparison_predictions_testingIndex.png", bbox_inches='tight',
            pad_inches=0.1)
plt.show()

fig, ax = plt.subplots()
ax.bar(x_pos-width, y_test[:, 1], width,
       capsize=10, label='Experimental Data', hatch='.')
ax.bar(x_pos,  y_predicted_383_new_full[:, 1], width,
       capsize=10, label='ANN Model predictions')
ax.bar(x_pos+width,  y_predicted_rvfl[:, 1], width,
       capsize=10, label='RVFL Model predictions')
ax.set_ylabel('Depth [µm]', fontsize=13)
ax.set_xlabel('Testing Index', fontsize=13)
ax.legend(fontsize=11)
ax.tick_params(axis='both', labelsize=13)
plt.savefig("depth_comparison_predictions_testingIndex.png", bbox_inches='tight',
            pad_inches=0.1)
plt.show()
