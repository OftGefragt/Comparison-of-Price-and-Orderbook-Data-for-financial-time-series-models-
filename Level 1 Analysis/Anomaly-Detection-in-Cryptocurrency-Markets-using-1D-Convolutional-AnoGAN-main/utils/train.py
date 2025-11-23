import torch
import config
from utils.discriminator_updated import Discriminator1D
from utils.encoder import Encoder1D
from utils.generator_updated import Generator1D
import torch.nn as nn
import torch.optim as optim
from utils.preprocessing import loader, F


G = Generator1D(z_dim=config.z_dim).to(config.device)
D = Discriminator1D().to(config.device)
E = Encoder1D(z_dim=config.z_dim, c_dim=F, seq_len=config.window_len).to(config.device)

#Checking the shapes
with torch.no_grad():
    z = torch.randn(4, config.z_dim, 1, device=config.device)
    g_out = G(z)
    print("G out", g_out.shape)   # expect [4, channels, seq_len]
    x_sample = next(iter(loader))[0][:4].to(config.device)
    z_e = E(x_sample)
    f_real, o_real = D(x_sample)
    print("E out", z_e.shape, "D feat", f_real.shape, "D out", o_real.shape)
    assert g_out.shape[1] == config.channels, "generator channels mismatch"
    assert g_out.shape[2] >= config.seq_len-4 and g_out.shape[2] <= config.seq_len+4, "check generator output length (may need layer tweaks)"

#Training the GAN
bce = nn.BCEWithLogitsLoss()
G_opt = optim.Adam(G.parameters(), lr=4e-4, betas=(0.5,0.999))
D_opt = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5,0.999))

for epoch in range(config.epochs_gan):
    for (real,) in loader:
        real = real.to(config.device)
        bs = real.size(0)
        # Train D
        D_opt.zero_grad()
        _, real_logits = D(real)
        real_labels = torch.ones(bs, device=config.device)
        loss_real = bce(real_logits, real_labels)
        z = torch.randn(bs, config.z_dim, 1, device=config.device)
        fake = G(z)
        _, fake_logits = D(fake.detach())
        fake_labels = torch.zeros(bs, device=config.device)
        loss_fake = bce(fake_logits, fake_labels)
        lossD = 0.5*(loss_real + loss_fake)
        lossD.backward(); D_opt.step()
        # Train G
        G_opt.zero_grad()
        _, fake_logits2 = D(fake)
        lossG = bce(fake_logits2, real_labels)
        lossG.backward(); G_opt.step()
    print(f"GAN Epoch {epoch+1}/{config.epochs_gan}  D {lossD.item():.4f}  G {lossG.item():.4f}")

#Training the encoder
E_opt = optim.Adam(E.parameters(), lr=1e-4, betas=(0.5,0.999))
mse = nn.MSELoss()
alpha = 1.0
beta = 0.1

for epoch in range(config.epochs_enc):
    for (real,) in loader:
        real = real.to(config.device)
        E_opt.zero_grad()
        # Encode real data
        z_enc = E(real)
        # Reconstruct via G
        rec = G(z_enc)
        # --- crop rec to match real length ---
        if rec.size(2) != real.size(2):
            rec = rec[:, :, :real.size(2)]
        # Feature matching
        f_real, _ = D(real)
        f_rec, _ = D(rec)
        loss_rec = mse(rec, real)
        loss_feat = mse(f_rec, f_real)
        lossE = alpha * loss_rec + beta * loss_feat
        lossE.backward()
        E_opt.step()
    print(f"Encoder Epoch {epoch+1}/{config.epochs_enc}  Loss {lossE.item():.4f}")

#Save models
torch.save(G.state_dict(), '../models/generator.pth')
torch.save(D.state_dict(), '../models/discriminator.pth')
torch.save(E.state_dict(), '../models/encoder.pth')