'''
Make sure to have enough data to pass convolutions, or you might want to alter the structure.
'''



class Generator1D(nn.Module):
    def __init__(self, z_dim=100, gf_dim=128, seq_len=64, channels=1):
        super().__init__()
        self.seq_len = seq_len

        self.model = nn.Sequential(
            nn.ConvTranspose1d(z_dim, gf_dim*4, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(gf_dim*4),
            nn.ELU(inplace=True),

            nn.ConvTranspose1d(gf_dim*4, gf_dim*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(gf_dim*2),
            nn.ELU(inplace=True),

            nn.ConvTranspose1d(gf_dim*2, gf_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(gf_dim),
            nn.ELU(inplace=True),

            nn.ConvTranspose1d(gf_dim, gf_dim//2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(gf_dim//2),
            nn.ELU(inplace=True),

            nn.ConvTranspose1d(gf_dim//2, gf_dim//4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(gf_dim//4),
            nn.ELU(inplace=True),

            nn.ConvTranspose1d(gf_dim//4, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.model(z)
