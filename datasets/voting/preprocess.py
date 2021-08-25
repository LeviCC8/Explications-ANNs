import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


data = []

data_map = {'republican': 0, 'democrat': 1,
            'n': 0, 'y': 1, '?': 2}


with open('house-votes-84.data', 'r') as f:
    for line in f:
        data_sample = line[:-1].split(',')
        data_sample = data_sample[1:] + [data_sample[0]]
        data.append([data_map[feature] for feature in data_sample])

data = np.array(data)
y = data[:, -1]
data = data[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=0,stratify=y)

data_train = np.append(X_train, np.expand_dims(y_train, 1), axis=1)
data_train = pd.DataFrame(data_train)

data_test = np.append(X_test, np.expand_dims(y_test, 1), axis=1)
data_test = pd.DataFrame(data_test)

data_train.to_csv('train.csv', index=False)
data_test.to_csv('test.csv', index=False)

print(len(data), len(data_train), len(data_test), len(data_train) + len(data_test))
