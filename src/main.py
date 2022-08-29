from utilities import load_data
import matplotlib.pyplot as plt
import numpy as np

X_train, y_train, X_test, y_test = load_data()

print('Train set images : ', X_train.shape)
print('Train set labels : ', y_train.shape)
print('Test set images  : ', X_test.shape)
print('Test set labels  : ', y_test.shape)


plt.show()
