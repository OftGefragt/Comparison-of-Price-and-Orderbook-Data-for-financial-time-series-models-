
def sliding_windows(arr, window=30, step=1):
    T, F = arr.shape
    if T < window:
        raise ValueError(f"time length {T} < window {window}")
    n = 1 + (T - window) // step
    out = np.stack([arr[i*step : i*step + window] for i in range(n)], axis=0)  # [N, window, F]
    return out  # [N, window, F]

windows = sliding_windows(arr, window=window_len, step=step)  # [N, window, channels]
# convert to [N, channels, seq_len]
X = windows.transpose(0,2,1).astype(np.float32)  # [N, channels, window_len]
print("windows shape", windows.shape, "X shape", X.shape)

scaler = MinMaxScaler(feature_range=(-1,1))
# flatten to [N*window_len, features]
flat = windows.reshape(-1, F)
flat_scaled = scaler.fit_transform(flat)
windows_scaled = flat_scaled.reshape(windows.shape)  # [N, window, F]
X_scaled = windows_scaled.transpose(0,2,1)  # [N, 14, window_len]

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

channels = 14 #chosen most important features -- adjust accordingly 
seq_len = window_len
#print("final tensor shape", X_tensor.shape, "channels", channels, "seq_len", seq_len)
