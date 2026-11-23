"""Final CFA/QSF shared feature and support operators."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from . import _support as own
from . import _alignment as legacy_cfa
sampling_grid=legacy_cfa.sampling_grid
warp=legacy_cfa.warp
pixel_centers=own.pixel_centers
normalize_coordinates=own.normalize_coordinates
DynamicSupport=own.DynamicSupport
combine=own.combine_factorized_logits

def cycle_gates(forward, backward, unit_hw, *, tau=2., epsilon=1e-3, detailed=False):
    """Same detached cycle arithmetic, optional CPU statistics outside hot path."""
    if tau <= 0 or epsilon <= 0:
        raise ValueError("Positive cycle scale and smoothing required")
    with torch.no_grad():
        f, b = forward.detach().float(), backward.detach().float()
        b_at_f, valid_f = warp(b, f, unit_hw)
        f_at_b, valid_b = warp(f, b, unit_hw)
        vb_at_f, _ = warp(valid_b.float(), f, unit_hw)
        vf_at_b, _ = warp(valid_f.float(), b, unit_hw)
        error_f, error_b = f+b_at_f, b+f_at_b
        _, end_f = sampling_grid(error_f, unit_hw)
        _, end_b = sampling_grid(error_b, unit_hw)
        distance_f = (error_f.square().sum(1, keepdim=True)+epsilon**2).sqrt()-epsilon
        distance_b = (error_b.square().sum(1, keepdim=True)+epsilon**2).sqrt()-epsilon
        mask_f = valid_f & end_f & vb_at_f.ge(1-1e-6)
        mask_b = valid_b & end_b & vf_at_b.ge(1-1e-6)
        gate_f, gate_b = torch.exp(-distance_f/tau)*mask_f, torch.exp(-distance_b/tau)*mask_b
        values = torch.stack((distance_f.mean(), distance_b.mean(), gate_f.mean(), gate_b.mean())).cpu().tolist() if detailed else [None]*4
    return gate_f, gate_b, dict(zip(("cycle_forward_mean_C1_pixels", "cycle_backward_mean_C1_pixels",
                                    "cycle_forward_gate_mean", "cycle_backward_gate_mean"), values))

@dataclass
class Features:
    levels: list
    context: torch.Tensor
    coarse: torch.Tensor

def extract(base, images):
    """Exactly one backbone, LightHam and original classifier evaluation."""
    levels = base.backbone(images)
    context = []
    hook = base.decode_head.align.register_forward_hook(lambda module, args, output: context.append(output))
    try:
        coarse = base.decode_head(levels)
    finally:
        hook.remove()
    if len(context) != 1:
        raise RuntimeError("One original Ham context required")
    return Features(list(levels), context[0], coarse)

def resize(value, size, evaluation=False):
    return F.interpolate(value.float() if evaluation else value, size=size, mode="bilinear", align_corners=False)

def zero_head(width, outputs):
    head = nn.Conv2d(width, outputs, 1)
    nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
    return head

def boundary_head(width):
    return nn.Sequential(nn.Conv2d(2*width, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))

class Projection(nn.Module):
    def __init__(self, c1_channels=64, context_channels=512, width=128):
        super().__init__()
        self.detail_project = nn.Conv2d(c1_channels, width, 1)
        self.context_project = nn.Conv2d(context_channels, width, 1)

    def forward(self, features, context):
        detail = self.detail_project(features[0])
        deep = self.context_project(context)
        original = resize(deep, detail.shape[-2:])
        return detail, deep, original

class Alignment(nn.Module):
    """Native-Ham/C1 bidirectional differentiable transfer, detached cycle gate."""
    def __init__(self, width=128, hidden=64, maximum=4.):
        super().__init__()
        self.forward_flow = legacy_cfa.flow_predictor(width, hidden)
        self.backward_flow = legacy_cfa.flow_predictor(width, hidden)
        self.maximum = float(maximum)
        self.detailed_diagnostics = False

    def forward(self, detail, deep):
        unit_hw = detail.shape[-2:]
        original = resize(deep, unit_hw)
        forward = self.maximum*self.forward_flow(torch.cat((detail, original), 1)).float().tanh()
        aligned, valid_f = warp(deep, forward, unit_hw)
        detail_down = resize(detail, deep.shape[-2:])
        backward = self.maximum*self.backward_flow(torch.cat((detail_down, deep), 1)).float().tanh()
        returned_deep, valid_b = warp(detail, backward, unit_hw)
        rho, rho_backward, diagnostics = cycle_gates(forward, backward, unit_hw, detailed=self.detailed_diagnostics)
        returned = resize(returned_deep*rho_backward, unit_hw)
        transfer = .5*(aligned*rho+returned)
        diagnostics = {**diagnostics, "flow_units": "C1_pixels_dx_dy", "maximum_displacement": self.maximum,
            "forward_valid_fraction": float(valid_f.float().mean().detach().cpu()) if self.detailed_diagnostics else None,
            "backward_valid_fraction": float(valid_b.float().mean().detach().cpu()) if self.detailed_diagnostics else None,
            "rho_detached": True, "rho_calibrated_correctness": False}
        return {"aligned": aligned, "transfer": transfer, "original": original,
                "forward": forward, "backward": backward, "rho": rho,
                "rho_backward": rho_backward, "valid_forward": valid_f,
                "valid_backward": valid_b, "diagnostic": diagnostics}

def support_forward(block, query, levels, points=None, attention_bias=None, self_value=None):
    """Sample original K/V once; optional absolute-C1 positions and self token.

    points: B,heads,3,4,H1,W1,2; attention_bias: B,heads,12,H1,W1.
    A self token uses the projected query as its key and supplied W-channel value.
    Every invalid external point remains masked; all-empty external sets use the
    original centre fallback only when self_value is absent.
    """
    batch, width, height, columns = query.shape
    if points is None:
        points, valid, _ = block.sampling_coordinates(query)
    else:
        if points.shape != (batch, block.heads, 3, 4, height, columns, 2):
            raise ValueError("Explicit support points must use the absolute C1 grid")
        if self_value is None:
            points, valid, _ = own.center_fallback(points, height, columns)
        else:
            valid = normalize_coordinates(points, height, columns).abs().le(1.).all(-1)
    per_head = width//block.heads
    keys, values = [], []
    for level, projector in enumerate(block.key_value):
        key_value = projector(levels[level])
        kh, kw = key_value.shape[-2:]
        key, value = key_value.chunk(2, 1)
        grid = normalize_coordinates(points[:, :, level], height, columns)
        grid = grid.permute(0, 1, 3, 4, 2, 5).reshape(batch*block.heads, height, columns*4, 2)
        sampled = []
        with torch.autocast(device_type=query.device.type, enabled=False):
            for source in (key, value):
                source = source.float().reshape(batch*block.heads, per_head, kh, kw)
                result = F.grid_sample(source, grid.float(), mode="bilinear", padding_mode="border", align_corners=False)
                result = result.view(batch, block.heads, per_head, height, columns, 4).permute(0, 1, 2, 5, 3, 4)
                sampled.append(result)
        keys.append(sampled[0]); values.append(sampled[1])
    key, value = torch.cat(keys, 3), torch.cat(values, 3)
    projected_query = block.query(query).float().view(batch, block.heads, per_head, height, columns)
    score = (projected_query.unsqueeze(3)*key).sum(2)/math.sqrt(per_head)
    if attention_bias is not None:
        if attention_bias.shape != score.shape:
            raise ValueError("Support attention prior shape differs")
        score = score+attention_bias.float()
    mask = valid.reshape(batch, block.heads, 12, height, columns)
    if self_value is not None:
        if self_value.shape != query.shape:
            raise ValueError("Self value must have the query grid and width")
        self_score = projected_query.square().sum(2, keepdim=True)/math.sqrt(per_head)
        score = torch.cat((self_score, score), 2)
        mask = torch.cat((torch.ones_like(mask[:, :, :1]), mask), 2)
        vself = self_value.float().view(batch, block.heads, per_head, 1, height, columns)
        value = torch.cat((vself, value), 3)
    weights = score.masked_fill(~mask, torch.finfo(score.dtype).min).softmax(2)
    aggregate = (weights.unsqueeze(2)*value).sum(3).reshape(batch, width, height, columns)
    output = block.output(aggregate)
    detailed = getattr(block, "detailed_diagnostics", False)
    with torch.no_grad():
        selected = int((weights.gt(0.) & mask).sum().cpu()) if detailed else None
        finite = bool(torch.isfinite(output).all()) if detailed else None
        self_mean = float(weights[:, :, 0].mean().cpu()) if detailed and self_value is not None else None
    return output, {"actual_selected_supports": selected, "finite": finite,
                    "active": selected > 0 and finite if detailed else None,
                    "activity_measured": detailed, "support_path_executed": True, "self_token": self_value is not None,
                    "self_attention_mean": self_mean, "source_features_resampled_once": True}
