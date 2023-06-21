import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
'''
folder = './Adap_baseline/'
y_pred = np.load(folder + 'oxford_pets-FS_2_16_y_pred.npy')
y_true = np.load(folder + 'oxford_pets-FS_2_16_y_true.npy')
y_true = (np.argmax(y_true, axis=1))

# print(y_true.shape)
plt.hist(y_pred, bins=37, rwidth=0.7)
plt.savefig(folder + 'pets_2_16_pred')
plt.close()
plt.hist(y_true, bins=37, rwidth=0.7)
plt.savefig(folder + 'pets_2_16_true')
plt.close()

cf_matrix = confusion_matrix(y_true, y_pred)
plt.imshow(cf_matrix)
plt.savefig(folder + 'cf_mat')
plt.close()
'''
index = 4
test = np.load('cf_mat/cf.npy')
get = test[index]
get[index] = 0
print(get / get.sum())
# print(cf_matrix)
# print(y_pred[0:200], y_true[0:200])