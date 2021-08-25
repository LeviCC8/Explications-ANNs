import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.model_selection import train_test_split


data = pd.read_csv('cleve.tsv', sep='\t')

all_cols = data.columns
y = data['target']
data.drop('target', axis=1, inplace=True)

# Normalize numeric features
numeric_cols = ['Age', 'Trestbps', 'Cholesterol', 'Max heart rate', 'Oldpeak']
scaler = MinMaxScaler()
scaler.fit(data[numeric_cols])
X = scaler.transform(data[numeric_cols])
X = pd.DataFrame(X)
X = scale(X)
data[numeric_cols] = X


X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=0,stratify=y)

data_train = np.append(X_train, np.expand_dims(y_train, 1), axis=1)
data_train = pd.DataFrame(data_train)

data_test = np.append(X_test, np.expand_dims(y_test, 1), axis=1)
data_test = pd.DataFrame(data_test)

data_train.to_csv('train.csv', index=False, header=all_cols)
data_test.to_csv('test.csv', index=False, header=all_cols)

print(len(data), len(data_train), len(data_test), len(data_train) + len(data_test))
