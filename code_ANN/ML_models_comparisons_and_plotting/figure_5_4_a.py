
import pandas as pd
import NNmodels
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

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

r_squared = r2_score(y_test, y_predicted_383_new_full)


plt.figure(1)
plt.scatter(y_test[:, 0], y_predicted_383_new_full[:, 0],
            marker='o', label="Diameter")
plt.scatter(y_test[:, 1], y_predicted_383_new_full[:, 1],
            marker='s', label="Depth")

plt.ylabel("Predicted values", fontsize=13)
plt.xlabel("Actual values", fontsize=13)
plt.text(90, 50, 'R-squared = %0.2f' % r_squared, fontsize=10)
print(r_squared)
plt.plot((np.array((0, 120))), (np.array((0, 120))), '--', color='k')
plt.legend(fontsize=11)
plt.tick_params(axis='both', labelsize=13)
plt.savefig("r2_383.png", bbox_inches='tight',
            pad_inches=0.1)
plt.show()
