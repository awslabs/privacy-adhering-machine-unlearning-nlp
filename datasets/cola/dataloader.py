import numpy as np
import os

pwd = os.path.dirname(os.path.realpath(__file__))

train_data = np.load(os.path.join(pwd, 'cola2_train.npy'), allow_pickle=True)
test_data = np.load(os.path.join(pwd, 'cola2_test.npy'), allow_pickle=True)

train_data = train_data.reshape((1,))[0]
test_data = test_data.reshape((1,))[0]
X_train = train_data["X"]
X_test = test_data["X"]
y_train = train_data['y'].astype(np.int64)
y_test = test_data['y'].astype(np.int64)

def load(indices, category='train'):
    res = []
    if category == 'train':
        for index in indices:
            res.append(X_train[index])
        return res, y_train[indices]
    elif category == 'test':
        for index in indices:
            res.append(X_test[index])
        return res, y_test[indices]