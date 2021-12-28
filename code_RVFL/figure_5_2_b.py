import numpy as np
import matplotlib.pyplot as plt


MSE_validation = [2.40E-03,	8.87E-03,	3.92E-03,
                  7.16E-03,	3.17E-03,	5.10E-03, 0]
MSE_test = [0, 0, 0, 0, 0, 0, 5.68E-03]
labels = ["Fold 0", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Average", "Test"]
x_pos = np.arange(len(labels))

width = 0.25
fig, ax = plt.subplots()
ax.bar(x_pos, MSE_validation, width,
       capsize=10, label='Validation', color='#ff7f0e')
ax.bar(x_pos, MSE_test, width,
       capsize=10, label='Test', color='#2ca02c')
ax.set_ylabel('MSE')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.legend()
plt.show()
