

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
z_dim = 100
gf_dim = 128
df_dim = 32
channels = 1

# Training parameters
epochs = 150
lr_G = 0.0002
lr_D = 0.0001
betas = (0.5, 0.999)

# Labels for label smoothing
real_label_val = 0.9
fake_label_val = 0.1

# Loss function
criterion = nn.BCELoss()

# Add noise function -- discriminator was overtaking generator in our training 
def add_noise(x, std=0.05):
    return x + torch.randn_like(x) * std

