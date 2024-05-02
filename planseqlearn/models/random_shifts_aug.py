import torch
import torch.nn.functional as F
from torch import nn


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
        self.last_shift = None

    def forward(self, x, repeat_last=False):
        n, _, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        if repeat_last:
            shift = self.last_shift
        else:
            shift = torch.randint(
                0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
            )
            shift *= 2.0 / (h + 2 * self.pad)
            self.last_shift = shift

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
