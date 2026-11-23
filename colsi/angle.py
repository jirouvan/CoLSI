"""Frozen ConvNeXt-Tiny ROI and 824-dimensional MoGe/RGB residual MLP."""
import os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
SCHEMA='colsi_frozen_T17_MoGe_fusion_v1'
EMBEDDING_DIM=768
AUX_DIM=7
RGB_ANGLE_INDEX=775
GEOMETRY_START=776

def build_rgb():
    from torchvision import models
    backbone=models.convnext_tiny(weights=None)
    backbone.classifier[-1]=nn.Identity()
    class ROIRegressor(nn.Module):
        def __init__(self):
            super().__init__();self.backbone=backbone
            self.head=nn.Sequential(nn.Linear(775,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
            self.register_buffer('aux_mean',torch.zeros(7));self.register_buffer('aux_std',torch.ones(7))
        def forward(self,images,aux):
            q=((aux-self.aux_mean)/self.aux_std).clamp(-10,10)
            return 90*torch.sigmoid(self.head(torch.cat((self.backbone(images),q),dim=1))[:,0])
    return ROIRegressor()

def load_angle_models(weights_dir,device='cuda:0'):
    from .runtime import weight_path
    rgb=build_rgb()
    payload=torch.load(weight_path(weights_dir,'convnext_tiny_roi.pt'),map_location='cpu',weights_only=True,mmap=True)
    rgb.load_state_dict(payload['model'],strict=True)
    rgb.to(device).float().eval().requires_grad_(False)
    payload=torch.load(weight_path(weights_dir,'fusion_mlp.pt'),map_location='cpu',weights_only=True)
    head=FusionResidualHead(payload['normalization'],variant='mlp')
    head.load_state_dict(payload['model'],strict=True)
    return rgb,head.to(device).float().eval().requires_grad_(False)

def require_device(device,*,allow_cpu_fixture=False):
    device=torch.device(device)
    if device.type=='cpu' and allow_cpu_fixture:return device
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('CUDA is required for neural inference/training')
    return device
def configure_fp32_runtime():
    """Explicitly preserve the original RGB numeric policy, never an AMP default."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

@torch.inference_mode()
def extract_rgb_features(model, images, aux, *, allow_cpu_fixture=False):
    """One backbone call gives both frozen embedding and original RGB prediction."""
    actual = images.device
    require_device(actual, allow_cpu_fixture=allow_cpu_fixture)
    if (images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224)
            or aux.shape != (len(images), AUX_DIM)
            or images.dtype != torch.float32 or aux.dtype != torch.float32
            or aux.device != actual or model.training
            or any(p.device != actual or p.requires_grad for p in model.parameters())):
        raise ValueError("Frozen/eval model and FP32 RGB[B,3,224,224], aux[B,7] required")
    if not torch.isfinite(images).all() or not torch.isfinite(aux).all():
        raise ValueError("Nonfinite ordinary RGB input cannot be silently repaired")
    with torch.autocast(actual.type, enabled=False):
        embedding = model.backbone(images)
        normalized_aux = ((aux - model.aux_mean) / model.aux_std).clamp(-10, 10)
        rgb_angle = 90 * torch.sigmoid(model.head(torch.cat((embedding, normalized_aux), dim=1))[:, 0])
    if embedding.shape != (len(images), EMBEDDING_DIM):
        raise ValueError("Frozen ConvNeXt embedding is not 768-dimensional")
    if embedding.device != actual or rgb_angle.device != actual or rgb_angle.dtype != torch.float32:
        raise RuntimeError("Actual frozen RGB computation must stay on the requested device")
    return {"embedding": embedding, "aux_normalized": normalized_aux,
            "rgb_angle_deg": rgb_angle}

def make_fusion_matrix(embedding, aux_normalized, rgb_angle_deg, moge_descriptor):
    arrays = [np.asarray(v, dtype=np.float32) for v in
              (embedding, aux_normalized, rgb_angle_deg, moge_descriptor)]
    e, q, alpha, geo = arrays
    if (e.ndim != 2 or e.shape[1] != EMBEDDING_DIM or q.shape != (len(e), AUX_DIM)
            or alpha.shape != (len(e),) or geo.ndim != 2 or len(geo) != len(e) or geo.shape[1] < 1):
        raise ValueError("Expected embedding[N,768], aux[N,7], alpha[N], geometry[N,D]")
    if not all(np.isfinite(v).all() for v in arrays):
        raise ValueError("Geometry must have explicit finite placeholders and validity flags")
    return np.ascontiguousarray(np.concatenate((e, q, alpha[:, None], geo), axis=1))

def fit_normalization(train_matrix):
    """Accept train rows only; there is deliberately no implicit split inference."""
    value = np.asarray(train_matrix, np.float64)
    if value.ndim != 2 or value.shape[1] <= GEOMETRY_START or not len(value) or not np.isfinite(value).all():
        raise ValueError("Finite train-only RGB+MoGe matrix required")
    mean, std = value.mean(0), value.std(0)
    std = np.maximum(std, 1e-4)
    return {"schema": SCHEMA + "_normalization", "mean": mean.tolist(), "std": std.tolist(),
            "fit_partition": "train", "fit_count": len(value), "std_floor": 1e-4,
            "normalized_clip_abs": 10., "geometry_start": GEOMETRY_START,
            "rgb_angle_index": RGB_ANGLE_INDEX}

class FusionResidualHead(nn.Module):
    """Unbounded learned correction; only final physical angle is clipped.

    Geometry-off is a same-weights mechanism diagnostic, not an independently
    trained RGB baseline. The original frozen RGB prediction is also returned.
    """
    def __init__(self, normalization, *, variant="mlp"):
        super().__init__()
        if (normalization.get("schema") != SCHEMA + "_normalization"
                or normalization.get("fit_partition") != "train"
                or normalization.get("geometry_start") != GEOMETRY_START
                or normalization.get("rgb_angle_index") != RGB_ANGLE_INDEX):
            raise ValueError("Current train-only fusion normalization contract required")
        mean = torch.as_tensor(normalization["mean"], dtype=torch.float32)
        std = torch.as_tensor(normalization["std"], dtype=torch.float32)
        if (mean.ndim != 1 or std.shape != mean.shape or len(mean) <= GEOMETRY_START
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not bool((std > 0).all())):
            raise ValueError("Finite positive train-only normalization required")
        if variant != "mlp":
            raise ValueError("The release uses the fixed MLP residual head")
        self.variant = variant
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.correction = nn.Sequential(nn.Linear(len(mean), 128), nn.ReLU(),
                                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        # Small nonzero initialization preserves the existing RGB estimate
        # while allowing geometry to influence predictions and gradients.
        last = self.correction[-1]
        nn.init.normal_(last.weight, std=1e-3)
        nn.init.zeros_(last.bias)

    def forward(self, matrix, *, geometry_enabled=True):
        if matrix.ndim != 2 or matrix.shape[1] != len(self.mean) or matrix.device != self.mean.device:
            raise ValueError("Fusion input shape/device differs from its frozen feature contract")
        normalized = ((matrix - self.mean) / self.std).clamp(-10, 10)
        if not geometry_enabled:
            normalized = torch.cat((normalized[:, :GEOMETRY_START],
                                    torch.zeros_like(normalized[:, GEOMETRY_START:])), dim=1)
        correction = self.correction(normalized)[:, 0]
        return matrix[:, RGB_ANGLE_INDEX] + correction

    @torch.inference_mode()
    def predict(self, matrix, *, geometry_enabled=True):
        raw = self(matrix, geometry_enabled=geometry_enabled)
        return {"raw_angle_deg": raw, "angle_deg": raw.clamp(0, 90),
                "valid": torch.isfinite(raw), "rgb_angle_deg": matrix[:, RGB_ANGLE_INDEX]}

def fit_fusion(train_matrix, train_targets_deg, *, variant="mlp", epochs=200,
               batch_size=1024, lr=1e-3, weight_decay=1e-4, seed=17,
               device="cuda:0", allow_cpu_fixture=False):
    """Fixed-budget fit. Root orchestration selects variants using validation only.

    Inputs must contain train rows only; this function accepts no dev/test
    labels and does not claim the resulting model improves the RGB baseline.
    """
    if seed != 17 or epochs < 1 or batch_size < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("Single seed17 and a positive fixed fit budget required")
    matrix = np.asarray(train_matrix, np.float32)
    target = np.asarray(train_targets_deg, np.float32)
    if target.shape != (len(matrix),) or not np.isfinite(target).all() or np.any((target < 0) | (target > 90)):
        raise ValueError("Finite train inclination targets in [0,90] required")
    normalization = fit_normalization(matrix)
    actual = require_device(device, allow_cpu_fixture=allow_cpu_fixture)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if actual.type == "cuda":
        configure_fp32_runtime()
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(actual)
        torch.cuda.synchronize(actual)
    start = time.perf_counter()
    model = FusionResidualHead(normalization, variant=variant).to(actual)
    x = torch.as_tensor(matrix, dtype=torch.float32, device=actual)
    y = torch.as_tensor(target, dtype=torch.float32, device=actual)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator(device=actual).manual_seed(seed)
    history = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(x), generator=generator, device=actual)
        total, count, updates = 0., 0, 0
        model.train()
        for ids in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(actual.type, enabled=False):
                prediction = model(x[ids])
                loss = torch.nn.functional.smooth_l1_loss(prediction / 10., y[ids] / 10., beta=.5)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite fusion loss; preserve the failed run")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            count += len(ids)
            updates += 1
        history.append({"epoch": epoch, "rows": count, "updates": updates, "loss": total / count})
    if actual.type == "cuda":
        torch.cuda.synchronize(actual)
    elapsed = time.perf_counter() - start
    model.eval()
    audit = {"schema": SCHEMA + "_fit", "seed": seed, "variant": variant,
             "epochs": epochs, "batch_size": batch_size, "train_rows": len(matrix),
             "updates": sum(v["updates"] for v in history), "seconds": elapsed,
             "row_exposures_per_second": len(matrix) * epochs / elapsed,
             "parameters": sum(p.numel() for p in model.parameters()),
             "device": str(actual), "precision": "fp32", "TF32": False,
             "train_only_normalization": True, "eval_targets_opened": False,
             "correction_range": "unbounded; final prediction clips only to physical [0,90]",
             "geometry_off_scope": "same-weights zero-standardized-geometry diagnostic",
             "cpu_fixture": bool(allow_cpu_fixture and actual.type == "cpu")}
    if actual.type == "cuda":
        audit.update(device_name=torch.cuda.get_device_name(actual),
                     dedicated_capacity_bytes=torch.cuda.get_device_properties(actual).total_memory,
                     owned_peak_allocated_bytes=torch.cuda.max_memory_allocated(actual),
                     owned_peak_reserved_bytes=torch.cuda.max_memory_reserved(actual))
    return model, {"normalization": normalization, "history": history, "audit": audit}
