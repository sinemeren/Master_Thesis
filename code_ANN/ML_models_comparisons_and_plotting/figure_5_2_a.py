import numpy as np
import matplotlib.pyplot as plt


MSE_train = [2.88E-03,	3.70E-03,	4.22E-03,	4.21E-03,	3.93E-03,	3.79E-03, 0]
MSE_validation = [3.14E-03,	8.14E-03,
                  5.18E-03,	6.19E-03,	5.44E-03,	5.62E-03, 0]
MSE_test = [0, 0, 0, 0, 0, 0, 4.11E-03]
labels = ["Fold 0", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Average", "Test"]
x_pos = np.arange(len(labels))

width = 0.25

fig, ax = plt.subplots()
ax.bar(x_pos-width, MSE_train, width,
       capsize=10, label='Training')
ax.bar(x_pos, MSE_validation, width,
       capsize=10, label='Validation')
ax.bar(x_pos, MSE_test, width,
       capsize=10, label='Test')
ax.set_ylabel('MSE', fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
plt.tick_params(axis='both', labelsize=12)
plt.legend(fontsize=13)
plt.show()
