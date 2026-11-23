"""QSF support sampling, factorized readout and boundary supervision."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def pixel_centers(height, width, *, device, dtype=torch.float32):
    yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                            torch.arange(width, device=device, dtype=dtype), indexing="ij")
    return torch.stack((xx, yy), dim=-1)

def normalize_coordinates(points, height, width):
    return 2.0 * (points + .5) / points.new_tensor((width, height)) - 1.0

def center_fallback(points, height, width):
    """Reject invalid samples, retaining C2/query centre if all12 are invalid."""
    valid = (normalize_coordinates(points, height, width).abs() <= 1.0).all(-1)
    fallback = ~valid.any(dim=(2, 3))
    first = torch.zeros_like(valid)
    first[:, :, 0, 0] = fallback
    centers = pixel_centers(height, width, device=points.device).view(1, 1, 1, 1, height, width, 2)
    return torch.where(first.unsqueeze(-1), centers, points), valid | first, fallback

def semantic_boundary(target):
    if target.ndim != 3:
        raise ValueError("Semantic targets must be BHW")
    valid = target.ne(255)
    edge = torch.zeros_like(valid)
    horizontal = (target[:, :, 1:] != target[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    vertical = (target[:, 1:, :] != target[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    edge[:, :, 1:] |= horizontal
    edge[:, :, :-1] |= horizontal
    edge[:, 1:, :] |= vertical
    edge[:, :-1, :] |= vertical
    return edge.unsqueeze(1).float(), valid.unsqueeze(1)

def boundary_loss(logits, target):
    labels, valid = semantic_boundary(target)
    score = F.interpolate(logits.float(), size=target.shape[-2:], mode="bilinear", align_corners=False)
    losses = F.binary_cross_entropy_with_logits(score, labels, reduction="none")
    return (losses * valid).sum() / valid.sum().clamp_min(1)

def combine_factorized_logits(coarse, foreground_delta, semantic_delta=None):
    """Restrict semantic correction to FG probability mass redistribution.

    This is not hard-support invariance or independent shared-backbone training.
    FP32 LSE then correction returned to incoming dtype preserves prior mixed
    precision order; exact mass identity is only approximate under BF16.
    """
    if coarse.ndim != 4 or coarse.shape[1] != 8:
        raise ValueError("BCHW background plus seven foreground logits required")
    if foreground_delta.shape != (coarse.shape[0], 1, *coarse.shape[-2:]):
        raise ValueError("Foreground correction must be B1HW")
    foreground = coarse[:, 1:]
    if semantic_delta is not None:
        if semantic_delta.shape != foreground.shape:
            raise ValueError("Seven-channel semantic correction shape differs")
        value, delta = foreground.float(), semantic_delta.float()
        change = torch.logsumexp(value + delta, dim=1, keepdim=True) - torch.logsumexp(value, dim=1, keepdim=True)
        foreground = foreground + (delta - change).to(dtype=coarse.dtype)
    foreground = foreground + foreground_delta.to(dtype=coarse.dtype)
    return torch.cat((coarse[:, :1], foreground), dim=1)

class DynamicSupport(nn.Module):
    """Self-written bounded 3-scale x4-point/head QKV support aggregation."""
    def __init__(self, level_channels, width=128, heads=4):
        super().__init__()
        if len(level_channels) != 3 or width % heads:
            raise ValueError("Three support levels and divisible attention width required")
        self.width, self.heads, self.level_count, self.samples = width, heads, 3, 4
        self.query = nn.Conv2d(width, width, 1)
        self.key_value = nn.ModuleList(nn.Conv2d(c, 2 * width, 1) for c in level_channels)
        self.output = nn.Conv2d(width, width, 1)
        self.coordinates = nn.Conv2d(width, heads * 3 * 4 * 2, 1)
        nn.init.zeros_(self.coordinates.weight)
        cardinal = torch.tensor(((1., 0.), (-1., 0.), (0., 1.), (0., -1.)))
        bias = (cardinal * .5).atanh().view(1, 1, 4, 2).expand(heads, 3, 4, 2).reshape(-1)
        with torch.no_grad():
            self.coordinates.bias.copy_(bias)

    def sampling_coordinates(self, feature):
        batch, _, height, width = feature.shape
        centers = pixel_centers(height, width, device=feature.device).view(1, 1, 1, 1, height, width, 2)
        raw = self.coordinates(feature).float().view(batch, self.heads, 3, 4, 2, height, width)
        offsets = 8.0 * raw.tanh().permute(0, 1, 2, 3, 5, 6, 4)
        return center_fallback(centers + offsets, height, width)

    def forward(self, feature, levels):
        batch, _, height, width = feature.shape
        points, valid, _ = self.sampling_coordinates(feature)
        keys, values = [], []
        per_head = self.width // self.heads
        for level, projector in enumerate(self.key_value):
            key_value = projector(levels[level])
            kh, kw = key_value.shape[-2:]
            key, value = key_value.chunk(2, dim=1)
            grid = normalize_coordinates(points[:, :, level], height, width)
            grid = grid.permute(0, 1, 3, 4, 2, 5).reshape(batch * self.heads, height, width * 4, 2)
            sampled = []
            with torch.autocast(device_type=feature.device.type, enabled=False):
                for source in (key, value):
                    source = source.float().reshape(batch * self.heads, per_head, kh, kw)
                    result = F.grid_sample(source, grid.float(), mode="bilinear", padding_mode="border", align_corners=False)
                    result = result.view(batch, self.heads, per_head, height, width, 4).permute(0, 1, 2, 5, 3, 4)
                    sampled.append(result)
            keys.append(sampled[0]); values.append(sampled[1])
        key, value = torch.cat(keys, dim=3), torch.cat(values, dim=3)
        query = self.query(feature).float().view(batch, self.heads, per_head, height, width)
        score = (query.unsqueeze(3) * key).sum(dim=2) / math.sqrt(per_head)
        mask = valid.reshape(batch, self.heads, 12, height, width)
        weights = score.masked_fill(~mask, torch.finfo(score.dtype).min).softmax(dim=2)
        aggregate = (weights.unsqueeze(2) * value).sum(dim=3).reshape(batch, self.width, height, width)
        return self.output(aggregate)
