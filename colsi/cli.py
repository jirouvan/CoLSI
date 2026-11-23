"""Minimal release commands; heavy neural imports occur after argument parsing."""
import argparse,json
from pathlib import Path
from .runtime import CODE_ROOT,DEFAULT_DATA,DEFAULT_WEIGHTS

def predict_main(argv=None):
    p=argparse.ArgumentParser(description='CoLSI-Net and the frozen MoGe/ConvNeXt/MLP landing pipeline')
    p.add_argument('--input',type=Path,required=True);p.add_argument('--camera',type=Path)
    p.add_argument('--weights-dir',type=Path,default=DEFAULT_WEIGHTS);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--segmentation-only',action='store_true');a=p.parse_args(argv)
    if not a.segmentation_only and a.camera is None:p.error('--camera is required for the full chain')
    import numpy as np
    from PIL import Image
    from .inference import Pipeline,save_frame
    from .runtime import write_json
    rgb=np.asarray(Image.open(a.input).convert('RGB'));pipeline=Pipeline(a.weights_dir,segmentation_only=a.segmentation_only)
    label,prob=pipeline.segment([rgb])[0]
    if a.segmentation_only:objects=pipeline.objects(rgb,None,None,label,prob,a.input.stem)
    else:
        camera=json.loads(a.camera.read_text('utf8'))
        objects=pipeline.objects(rgb,camera['K'],camera['g_C'],label,prob,a.input.stem)
    save_frame(a.output,a.input.stem,label,objects);write_json(a.output/'device.json',pipeline.audit)
    print(str(a.output/'prediction.json'))

def train_main(argv=None):
    p=argparse.ArgumentParser(description='Train the final CoLSI-Net architecture on the original canonical split')
    p.add_argument('--data-root',type=Path,default=DEFAULT_DATA);p.add_argument('--weights-dir',type=Path,default=DEFAULT_WEIGHTS)
    p.add_argument('--config',type=Path,default=CODE_ROOT/'configs/train.json');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args(argv)
    from .training import train
    train(a.data_root,a.weights_dir,a.output,config_path=a.config,resume=a.resume)

def evaluate_main(argv=None):
    p=argparse.ArgumentParser(description='Canonical B8 segmentation or complete-task evaluation')
    p.add_argument('--data-root',type=Path,default=DEFAULT_DATA);p.add_argument('--weights-dir',type=Path,default=DEFAULT_WEIGHTS)
    p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--task',choices=['segmentation','full-chain'],default='segmentation')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    from .evaluation import evaluate
    result=evaluate(a.data_root,a.weights_dir,a.output,split=a.split,task=a.task)
    print(json.dumps(result,ensure_ascii=False))
