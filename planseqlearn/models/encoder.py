from torch import nn

from planseqlearn import utils


class DrQV2Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class PoolEncoder(nn.Module):
    def __init__(self, obs_shape, repr_dim=None):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),  # 41 x 41
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),  # 39 x 39
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),  # 37 x 37
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),  # 35 x 35
            nn.ReLU(),
            nn.AvgPool2d(4, stride=4),  # 32 * 8 * 8
        )

        if repr_dim is None:
            self.repr_dim = 32 * 8 * 8
            self.projection = nn.Identity()
        else:
            self.repr_dim = repr_dim
            self.projection = nn.Sequential(
                nn.Linear(32 * 8 * 8, repr_dim),
            )
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        h = self.projection(h)
        return h
