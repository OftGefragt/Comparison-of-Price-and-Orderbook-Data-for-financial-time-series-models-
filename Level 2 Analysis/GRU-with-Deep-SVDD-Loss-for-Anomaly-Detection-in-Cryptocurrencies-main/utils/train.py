input_dim = X.shape[2]  # number of features per timestep
embedding_dim = 256

#using a large embedding dim can help you reduce bias 

gru_encoder = GRUEncoder(input_dim=input_dim, embedding_dim=embedding_dim, n_layers=2, dropout=0.2)

#We get the optimum c and save it.

c = train_deep_svdd(gru_encoder, X_train, epochs=20, batch_size=64, lr=1e-3, lr_c=0.01, device='cpu')
#you might want to change your device to cuda if you have it, for faster training 

torch.save(c, 'svdd_center.pth')

# Save GRU encoder weights
torch.save(gru_encoder.state_dict(), 'gru_encoder.pth')


# Save scaler for future new data
import joblib
joblib.dump(scaler, 'scaler.save')
