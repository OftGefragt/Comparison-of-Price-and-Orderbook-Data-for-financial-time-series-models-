import torch
import matplotlib.pyplot as plt

def compute_anomaly_scores(gru_encoder, X_new, c, device='cpu'):
    gru_encoder.to(device)
    X_new = X_new.to(device)
    c = c.to(device)

    with torch.no_grad():
        embeddings = gru_encoder(X_new)
        diff = embeddings - c
        scores = torch.sum(diff**2, dim=1)

    return scores

def detect_anomalies(anomaly_scores, percentile=95):
    threshold = torch.quantile(anomaly_scores, percentile / 100.0)
    anomalies = (anomaly_scores > threshold)
    return anomalies, threshold

def plot_anomaly_scores(anomaly_scores, anomalies, threshold):
    scores_np = anomaly_scores.cpu().numpy()
    anomalies_np = anomalies.cpu().numpy()
    threshold_np = threshold.cpu().numpy()

    plt.figure(figsize=(12, 5))
    
    normal_indices = ~anomalies_np
    anomaly_indices = anomalies_np
    
    plt.scatter(np.where(normal_indices)[0], scores_np[normal_indices], 
                c='blue', label='Normal', alpha=0.6)
    plt.scatter(np.where(anomaly_indices)[0], scores_np[anomaly_indices], 
                c='red', label='Anomaly', alpha=0.8)
    
    plt.axhline(threshold_np, color='black', linestyle='--', label='Threshold')
    plt.xlabel("Sequence Index")
    plt.ylabel("Anomaly Score")
    plt.title("Deep SVDD Anomaly Detection Results")
    plt.legend()
    plt.tight_layout()
    plt.show()

#getting the scores
anomaly_scores = compute_anomaly_scores(gru_encoder, X_test, c, device='cpu')


anomalies, threshold = detect_anomalies(anomaly_scores, percentile=95)

