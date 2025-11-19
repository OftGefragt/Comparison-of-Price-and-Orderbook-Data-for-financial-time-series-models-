from model.gru import GRUEncoder, train_deep_svdd
from utils.inference import compute_anomaly_scores, detect_anomalies, plot_anomaly_scores, plot_anomaly_context
from utils.prior import X_train, X_test, scaler, time_step, test_data
import torch

input_dim = X_train.shape[2]  # number of features per timestep
embedding_dim = 256

#using a large embedding dim can help you reduce bias 

gru_encoder = GRUEncoder(input_dim=input_dim, embedding_dim=embedding_dim, n_layers=2, dropout=0.2)

#We get the optimum c and save it.

c = train_deep_svdd(gru_encoder, X_train, epochs=2, batch_size=64, lr=1e-3, lr_c=0.001, device='mps')
#you might want to change your device to cuda if you have it, for faster training 

torch.save(c, 'svdd_center.pth')

# Save GRU encoder weights
torch.save(gru_encoder.state_dict(), 'gru_encoder.pth')


# Save scaler for future new data
import joblib
joblib.dump(scaler, 'scaler.save')

percentile = 95

#getting the scores
anomaly_scores = compute_anomaly_scores(gru_encoder, X_test, c, device='cpu')

anomalies, threshold = detect_anomalies(anomaly_scores, percentile=95)

top_k = 10

top_scores, top_indices = torch.topk(anomaly_scores, k=top_k)

for i in range(top_k):
    idx = top_indices[i].item()
    score = top_scores[i].item()
    timestamp_idx = idx + time_step-1

    if timestamp_idx < len(X_test):
        anomaly_timestamp = test_data.index[timestamp_idx]
        print(f"Anomaly {i+1}: Timestamp: {anomaly_timestamp}, Anomaly Score: {score:.8f}")
        plot_anomaly_context(test_data, anomaly_timestamp, window_minutes=30)
    else:
        print(f"Anomaly {i+1}: Anomaly Score: {score:.8f}, Index out of bounds for timestamp retrieval.")

print(f"Plotting results with a threshold at the {percentile}th percentile...")
plot_anomaly_scores(anomaly_scores, anomalies, threshold)