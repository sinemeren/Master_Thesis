
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

# random set number
randNum = 1

y_predicted_383_new_full = NNmodels.predict("383_new_full")


if(randNum == 1):
    # set 1
    y_predicted_383_new_full_20 = NNmodels.predict(
        "383_new_full_20")
    y_predicted_383_new_full_30 = NNmodels.predict(
        "383_new_full_30")
    y_predicted_383_new_full_40 = NNmodels.predict(
        "383_new_full_40")
    y_predicted_383_new_full_50 = NNmodels.predict(
        "383_new_full_50")
    y_predicted_383_new_full_60 = NNmodels.predict(
        "383_new_full_60")
elif(1 < randNum and randNum <= 4):

    y_predicted_383_new_full_20 = NNmodels.predict(
        "383_new_full_20" + "_" + str(randNum))
    y_predicted_383_new_full_30 = NNmodels.predict(
        "383_new_full_30" + "_" + str(randNum))
    y_predicted_383_new_full_40 = NNmodels.predict(
        "383_new_full_40" + "_" + str(randNum))
    y_predicted_383_new_full_50 = NNmodels.predict(
        "383_new_full_50" + "_" + str(randNum))
    y_predicted_383_new_full_60 = NNmodels.predict(
        "383_new_full_60" + "_" + str(randNum))


ae_diameter_predicted383_20, ae_depth_predicted383_20 = abs(
    y_test[:, 0] - y_predicted_383_new_full_20[:, 0]), abs(y_test[:, 1] - y_predicted_383_new_full_20[:, 1])

ae_diameter_predicted383_30, ae_depth_predicted383_30 = abs(
    y_test[:, 0] - y_predicted_383_new_full_30[:, 0]), abs(y_test[:, 1] - y_predicted_383_new_full_30[:, 1])

ae_diameter_predicted383_40, ae_depth_predicted383_40 = abs(
    y_test[:, 0] - y_predicted_383_new_full_40[:, 0]), abs(y_test[:, 1] - y_predicted_383_new_full_40[:, 1])

ae_diameter_predicted383_50, ae_depth_predicted383_50 = abs(
    y_test[:, 0] - y_predicted_383_new_full_50[:, 0]), abs(y_test[:, 1] - y_predicted_383_new_full_50[:, 1])

ae_diameter_predicted383_60, ae_depth_predicted383_60 = abs(
    y_test[:, 0] - y_predicted_383_new_full_60[:, 0]), abs(y_test[:, 1] - y_predicted_383_new_full_60[:, 1])


ae_diameter_predicted383_full, ae_depth_predicted383_full = abs(
    y_test[:, 0] - y_predicted_383_new_full[:, 0]), abs(y_test[:, 1] - y_predicted_383_new_full[:, 1])


x_pos = np.array([20, 30, 40, 50, 60, 73])


args_diameter = (ae_diameter_predicted383_20, ae_diameter_predicted383_30,
                 ae_diameter_predicted383_40, ae_diameter_predicted383_50,
                 ae_diameter_predicted383_60, ae_diameter_predicted383_full)

data_diameter = np.concatenate(args_diameter)

args_depth = (ae_depth_predicted383_20, ae_depth_predicted383_30, ae_depth_predicted383_40,
              ae_depth_predicted383_50, ae_depth_predicted383_60, ae_depth_predicted383_full)

data_depth = np.concatenate(args_depth)

mlmodels = []
j = 0
for i in range((len(x_pos)*len(ae_depth_predicted383_20))):

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
plt.ylim([-1, 40])
plt.savefig("boxPlot_NNModel_383_differentNumOfData_rand_" + str(randNum) + ".png", bbox_inches='tight',
            pad_inches=0.1)
plt.show()
