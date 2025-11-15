
class Discriminator1D(nn.Module):
    def __init__(self, c_dim=1, df_dim=32):
        super().__init__()
        self.conv0 = nn.Conv1d(c_dim, df_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.elu0 = nn.ELU(inplace=True)

        self.conv1 = nn.Conv1d(df_dim, df_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(df_dim*2)
        self.elu1 = nn.ELU(inplace=True)

        self.conv2 = nn.Conv1d(df_dim*2, 1, kernel_size=4, stride=1, padding=0, bias=False)

        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h0 = self.elu0(self.conv0(x))
        h1 = self.elu1(self.bn1(self.conv1(h0)))
        h2 = self.conv2(h1)
        out = torch.mean(h2, dim=2)
        return h1, torch.sigmoid(out).view(-1)
