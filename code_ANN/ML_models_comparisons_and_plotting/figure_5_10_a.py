
import pandas as pd
import NNmodels_20Analysis
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


y_predicted_20_Analysis_set1, ae_predictions_1 = NNmodels_20Analysis.predict(
    "20_data_points_set1")

y_predicted_20_Analysis_set2, ae_predictions_2 = NNmodels_20Analysis.predict(
    "20_data_points_set2")

y_predicted_20_Analysis_set3, ae_predictions_3 = NNmodels_20Analysis.predict(
    "20_data_points_set3")

y_predicted_20_Analysis_set4, ae_predictions_4 = NNmodels_20Analysis.predict(
    "20_data_points_set4")


ae_diameter_predicted383_20_set_1, ae_depth_predicted383_20_set_1 = ae_predictions_1[
    :, 0], ae_predictions_1[:, 1]

ae_diameter_predicted383_20_set_2, ae_depth_predicted383_20_set_2 = ae_predictions_2[
    :, 0], ae_predictions_2[:, 1]
ae_diameter_predicted383_20_set_3, ae_depth_predicted383_20_set_3 = ae_predictions_3[
    :, 0], ae_predictions_3[:, 1]
ae_diameter_predicted383_20_set_4, ae_depth_predicted383_20_set_4 = ae_predictions_4[
    :, 0], ae_predictions_4[:, 1]

x_pos = np.array([1, 2, 3, 4])


args_diameter = (ae_diameter_predicted383_20_set_1, ae_diameter_predicted383_20_set_2,
                 ae_diameter_predicted383_20_set_3, ae_diameter_predicted383_20_set_4)

data_diameter = np.concatenate(args_diameter)

args_depth = (ae_depth_predicted383_20_set_1, ae_depth_predicted383_20_set_2,
              ae_depth_predicted383_20_set_3, ae_depth_predicted383_20_set_4)

data_depth = np.concatenate(args_depth)


sets = []
j = 0
for i in range((len(x_pos)*len(ae_diameter_predicted383_20_set_1))):

    if(i % 61 == 0 and i > 60):
        j = j+1
    sets.append(x_pos[j])

df = pd.DataFrame(
    {'Random Sets': sets, 'Diameter': data_diameter, 'Depth': data_depth})

dd = pd.melt(df, id_vars=['Random Sets'], value_vars=[
             'Diameter', 'Depth'], var_name='Predicted values')
sns.boxplot(x='Random Sets', y='value',
            data=dd, hue='Predicted values')

plt.ylabel("Absolute Error in µm")
plt.savefig("boxPlot_NNModel_383_differentNumOfData_train_20_test_61.png", bbox_inches='tight',
            pad_inches=0.1)
plt.show()
