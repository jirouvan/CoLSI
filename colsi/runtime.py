"""Explicit CUDA setup shared by the release entry points."""
import json
import os
from pathlib import Path

CODE_ROOT=Path(__file__).resolve().parents[1]
DEFAULT_DATA=CODE_ROOT.parent/'CoLSI-data'/'CoLSI'
DEFAULT_WEIGHTS=CODE_ROOT.parent/'CoLSI-data'/'weights'

def configure_cuda(seed=17,threads=2):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import random,numpy as np,torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required; install a matching CUDA-enabled PyTorch build')
    torch.set_num_threads(threads)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    return torch.device('cuda:0')

def device_report(model=None,inputs=None):
    import torch
    value={'device':'cuda:0','name':torch.cuda.get_device_name(0),
      'dedicated_capacity_bytes':torch.cuda.get_device_properties(0).total_memory,
      'torch':torch.__version__,'cuda':torch.version.cuda,'TF32':False,
      'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),
      'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    if model is not None:value['model_devices']=sorted({str(p.device) for p in model.parameters()})
    if inputs is not None:value['input_device']=str(inputs.device);value['input_shape']=list(inputs.shape)
    return value

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def weight_path(folder,name):
    result=Path(folder)/name
    if not result.is_file():raise FileNotFoundError(f'Place {name} in {Path(folder)}; see the release weights manifest')
    return result
