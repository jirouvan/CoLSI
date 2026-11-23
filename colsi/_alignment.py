"""CFA sampling and learned bidirectional displacement primitives."""
import torch
from torch import nn
from torch.nn import functional as F

def sampling_grid(flow, unit_hw):
    """Flow grid is its destination grid; displacement unit is always C1 px."""
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("Flow must be B2HW, ordered dx/dy")
    hu, wu = map(int, unit_hw)
    if hu < 1 or wu < 1:
        raise ValueError("Positive common C1 coordinate units required")
    _, _, height, width = flow.shape
    yy, xx = torch.meshgrid(torch.arange(height, device=flow.device, dtype=torch.float32),
                            torch.arange(width, device=flow.device, dtype=torch.float32), indexing="ij")
    origin = torch.stack((2 * (xx + .5) / width - 1,
                          2 * (yy + .5) / height - 1), dim=-1)[None]
    delta = flow.float().permute(0, 2, 3, 1) * flow.new_tensor((2. / wu, 2. / hu), dtype=torch.float32)
    grid = origin + delta
    valid = grid.abs().le(1.).all(dim=-1).unsqueeze(1)
    return grid, valid

def warp(source, flow, unit_hw):
    """align_false cell-centre sampling, border extension, explicit valid mask."""
    if source.ndim != 4 or source.shape[0] != flow.shape[0]:
        raise ValueError("BCHW source/flow batches must match")
    grid, valid = sampling_grid(flow, unit_hw)
    with torch.autocast(device_type=source.device.type, enabled=False):
        value = F.grid_sample(source.float(), grid.float(), mode="bilinear",
                              padding_mode="border", align_corners=False)
    return value, valid

def flow_predictor(width, hidden=64):
    block = nn.Sequential(nn.Conv2d(2 * width, hidden, 3, padding=1), nn.GELU(),
                          nn.Conv2d(hidden, 2, 3, padding=1))
    nn.init.zeros_(block[-1].weight); nn.init.zeros_(block[-1].bias)
    return block
