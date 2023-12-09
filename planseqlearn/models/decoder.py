from torch import nn

from planseqlearn import utils


class SacAeDecoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, output_act=nn.Identity(), repr_dim=None
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_act = output_act
        if repr_dim is None:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Sequential(
                nn.Linear(repr_dim, in_channels * 35 * 35), nn.ReLU()
            )

        self.transpose_convnet = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 3, 1),  # 37 x 37
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, 1),  # 39 x 39
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, 1),  # 41 x 41
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 3, 2, 0, 1),
        )

        self.apply(utils.weight_init)

    def forward(self, x):
        out = self.projection(x)
        out = out.view(out.shape[0], self.in_channels, 35, 35)
        out = self.transpose_convnet(out)
        out = self.output_act(out)
        return out


class PoolDecoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, output_act=nn.Identity(), repr_dim=None
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_act = output_act
        if repr_dim is None:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Sequential(
                nn.Linear(repr_dim, in_channels * 8 * 8), nn.ReLU()
            )

        self.transpose_convnet = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 16, 4, 4, 0, 3),  # 35 x 35
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, 3, 1),  # 37 x 37
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, 1),  # 39 x 39
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, 1),  # 41 x 41
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 3, 2, 0, 1),
        )

        self.apply(utils.weight_init)

    def forward(self, x):
        out = self.projection(x)
        out = out.view(out.shape[0], self.in_channels, 8, 8)
        out = self.transpose_convnet(out)
        out = self.output_act(out)
        return out
