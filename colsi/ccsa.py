"""CCSA: final CFA/QSF consistency-guided cross-scale support attention."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from . import _fusion as c
VARIANTS=('f05',)
def candidate_cycle_prior(rho, points):
    """Detached candidate-location prior, B,heads,12,H1,W1.

    C1/Ham rho is extrapolated to multiscale sample locations. This is an
    observable heuristic, not independently calibrated per-level matching GT.
    """
    if points.ndim != 7 or points.shape[2:4] != (3, 4) or points.shape[-1] != 2:
        raise ValueError('Absolute C1 three-by-four support coordinates required')
    batch, heads, _, _, height, width, _ = points.shape
    if rho.shape != (batch, 1, height, width):
        raise ValueError('C1 consistency map required')
    with torch.no_grad(), torch.autocast(device_type=rho.device.type, enabled=False):
        grid = c.normalize_coordinates(points.detach().float(), height, width)
        valid = grid.abs().le(1).all(-1)
        grid = grid.permute(0, 1, 4, 5, 2, 3, 6).reshape(batch * heads, height, width * 12, 2)
        source = rho.detach().float().expand(-1, heads, -1, -1).reshape(batch * heads, 1, height, width)
        values = F.grid_sample(source, grid, mode='bilinear', padding_mode='border', align_corners=False)
        values = values.view(batch, heads, height, width, 12).permute(0, 1, 4, 2, 3)
        return values.clamp(0, 1) * valid.reshape(batch, heads, 12, height, width)

def _fuse(channels, width):
    return nn.Sequential(nn.Conv2d(channels, width, 1), nn.GELU())

class FusionBlock(nn.Module):

    def __init__(self, variant, c1_channels=64, context_channels=512, level_channels=(128, 320, 512), width=128, heads=4, flow_hidden=64, **options):
        super().__init__()
        if options:
            raise ValueError('Unknown extension options: ' + str(sorted(options)))
        self.variant, self.width, self.heads = (variant, width, heads)
        self.detailed_diagnostics = False
        self.projection = c.Projection(c1_channels, context_channels, width)
        self.alignment = c.Alignment(width, flow_hidden, maximum=4.0)
        self.support = c.DynamicSupport(level_channels, width=width, heads=heads)
        self.boundary = c.boundary_head(width)
        self.quality = c.zero_head(width, 1)
        self.semantic = c.zero_head(width, 7)
        self.fuse = _fuse(2 * width, width)
        self.condition = self.gate = self.edge_fuse = None
        self.last_diagnostic = None

    def forward(self, features, context, *, anchor_logits=None, evaluation=False):
        if len(features) != 4:
            raise ValueError('One shared C1/C2/C3/C4 feature pass required')
        detail, deep, original = self.projection(features, context)
        boundary = self.boundary(torch.cat((detail, original), 1))
        alignment = None if self.alignment is None else self.alignment(detail, deep)
        diagnostic = {'variant': self.variant, 'single_boundary_head': True, 'boundary_total_weight': 0.1, 'rho_is_calibrated_probability': False, 'coarse_flow_units': 'C1_pixels_dx_dy', 'original_multiscale_support_library': True, 'evaluation': bool(evaluation)}
        query = self.fuse(torch.cat((detail, alignment['transfer']), 1))
        points, _, _ = self.support.sampling_coordinates(query)
        prior = candidate_cycle_prior(alignment['rho'], points)
        bias = prior.clamp_min(0.0001).log()
        aggregate, stats = c.support_forward(self.support, query, features[1:], points=points, attention_bias=bias, self_value=query)
        refined = query + aggregate
        prior_min, prior_max = torch.stack((prior.min(), prior.max())).detach().cpu().tolist() if self.detailed_diagnostics else (None, None)
        diagnostic.update(candidate_prior_min=prior_min, candidate_prior_max=prior_max, self_null_candidates=1, external_candidates=12, candidate_prior_detached=True, reverse_flow_learning_path='bidirectional_transfer_to_query_and_values', prior_source='candidate_locations_of_C1_Ham_rho_extrapolated_not_per_scale_truth')
        quality, semantic = (self.quality(refined), self.semantic(refined))
        finite = active = None
        if self.detailed_diagnostics:
            with torch.no_grad():
                finite = bool(torch.stack([torch.isfinite(x).all() for x in (refined, quality, semantic, boundary)]).all())
            active = bool(stats['active']) and finite
        diagnostic.update(support=stats, finite=finite, active=active, activity_measured=self.detailed_diagnostics, support_path_executed=True, safety_checks='engine_actual_loss_and_gradients')
        if alignment is not None:
            diagnostic['alignment'] = alignment['diagnostic']
        self.last_diagnostic = diagnostic
        return {'quality': quality, 'semantic': semantic, 'boundary': boundary, 'feature': refined, 'diagnostic': diagnostic}
