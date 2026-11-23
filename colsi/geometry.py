"""Frozen MoGe-3 ViT-L native-camera frontend."""
from pathlib import Path
import math,sys
import numpy as np
from .runtime import CODE_ROOT

def setup_vendor():
    for name in ['MoGe','FlexGEMM','utils3d-moge']:
        path=str(CODE_ROOT/'third_party'/name)
        if path not in sys.path:sys.path.insert(0,path)

def load_moge(checkpoint,device='cuda:0'):
    setup_vendor()
    import torch
    from moge.model.v3 import MoGeModel
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if payload['model_config'].get('refiner') is None:raise ValueError('MoGe-3 trained refiner configuration is required')
    model=MoGeModel(**payload['model_config'])
    model.load_state_dict(payload['model'],strict=True)
    for module in model.modules():
        if type(module).__module__.startswith('flex_gemm') and hasattr(module,'allow_tf32'):module.allow_tf32=False
    return model.to(device).eval().requires_grad_(False)

def predict_geometry(model,rgb,K,device='cuda:0'):
    import torch
    h,w=rgb.shape[:2];K=np.asarray(K,np.float64)
    if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0:raise ValueError('Finite positive-focal K is required')
    if not np.isclose(K[0,0],K[1,1],rtol=1e-8) or not np.allclose(K[:2,2],[(w-1)/2,(h-1)/2],atol=1e-7):
        raise ValueError('This frozen FOV adapter requires square pixels and a centered principal point')
    x=torch.from_numpy(rgb.copy()).permute(2,0,1).float().to(device)/255
    fov=math.degrees(2*math.atan(w/(2*K[0,0])))
    with torch.inference_mode():
        result=model.infer(x,fov_x=fov,resolution_level=9,refine_steps=3,use_fp16=False,apply_mask=False,force_projection=True)
    z=result['depth'].float().cpu().numpy()
    normal=result['normal'].float().cpu().numpy()
    mask=result.get('mask')
    valid_z=np.isfinite(z)&(z>0)
    valid_n=np.isfinite(normal).all(-1)&(np.linalg.norm(normal,axis=-1)>1e-8)
    if mask is not None:
        mask=mask.cpu().numpy().astype(bool);valid_z&=mask;valid_n&=mask
    if z.shape!=(h,w) or normal.shape!=(h,w,3):raise ValueError('Native MoGe output shape mismatch')
    if not valid_z.any():raise ValueError('MoGe produced no finite positive valid depth')
    return {'axial_z':np.where(valid_z,z,np.nan).astype(np.float32),
      'normal_direct':np.where(valid_n[...,None],normal,np.nan).astype(np.float32),
      'valid_depth':valid_z,'valid_normal':valid_n}
