import sys

#do your uploads
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#prepare your sequneces
df = pd.read_csv('../data/btcusd_1-min_data_reduced.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df.set_index('Timestamp', inplace=True)

df.dropna(inplace=True)
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
df_features = df[feature_columns].copy()


#df = df.iloc[: , :30]

split_ratio = 0.8
num_train = int(len(df_features) * split_ratio)
train_data = df_features[:num_train]
test_data = df_features[num_train:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

def create_sequences(data, time_step=10):
    X = []
    for i in range(len(data) - time_step + 1):
        X.append(data[i:i+time_step])
    return np.array(X)

time_step = 10
X_train = create_sequences(scaled_train_data, time_step=time_step)
X_test = create_sequences(scaled_test_data, time_step=time_step)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

print(X_train.shape)
print(X_test.shape)

"""

X = create_sequences(scaled_data, time_step=time_step)

split_ratio = 0.8
num_train = int(len(X) * split_ratio)
X_train = X[:num_train]
X_test = X[num_train:]

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
"""