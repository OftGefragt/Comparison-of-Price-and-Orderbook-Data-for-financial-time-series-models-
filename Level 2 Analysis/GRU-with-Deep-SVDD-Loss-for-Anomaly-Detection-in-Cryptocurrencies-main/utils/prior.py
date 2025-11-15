
#do your uploads 
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#prepare your sequneces
df = pd.read_csv('/content/df_selection.csv')
df = df.iloc[: , :30]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

def create_sequences(data, time_step=10):
    X = []
    for i in range(len(data) - time_step + 1):
        X.append(data[i:i+time_step])
    return np.array(X)

time_step = 10
X = create_sequences(scaled_data, time_step=time_step)

split_ratio = 0.8
num_train = int(len(X) * split_ratio)
X_train = X[:num_train]
X_test = X[num_train:]

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
