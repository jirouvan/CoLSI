"""Portable canonical CoLSI-Sim records and visible-semantic training data."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .segmentation import prepare

def records(data_root,split,*,supervision=False):
    root=Path(data_root);sim=root/'CoLSI-Sim'
    fixed=json.loads((root/'benchmark/splits/sim_canonical_split.json').read_text('utf-8-sig'))['splits'][split]
    path=sim/'samples.jsonl' if supervision else sim/'input/samples.jsonl'
    wanted=set(fixed);byid={}
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            row=json.loads(line)
            if row['sample_id'] in wanted:
                if row['split']!=split or row['sample_id'] in byid:raise ValueError('Partition/duplicate record mismatch')
                byid[row['sample_id']]=row
    if set(byid)!=wanted:raise ValueError('Canonical partition population differs')
    return [byid[sid] for sid in fixed]

def read_image(root,path,expected_sha256=None,*,rgb=False):
    file=(Path(root)/path).resolve()
    if not file.is_relative_to(Path(root).resolve()):raise ValueError('Data path escapes its root')
    if expected_sha256:
        with file.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected_sha256:raise ValueError('Image identity differs: '+path)
    with Image.open(file) as image:return np.array(image.convert('RGB') if rgb else image,copy=True)

class SegmentationDataset(Dataset):
    def __init__(self,data_root,split):
        self.root=Path(data_root)/'CoLSI-Sim';self.rows=records(data_root,split,supervision=True)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        rgb=read_image(self.root,row['rgb']['path'],row['rgb']['sha256'],rgb=True)
        target=read_image(self.root,row['semantic_mask']['path'],row['semantic_mask']['sha256'])
        x,y,_=prepare(rgb,target,(288,512))
        return {'image':x,'target':y,'native_target':target,'sample_id':row['sample_id']}

def collate(rows):
    import torch
    return {'image':torch.stack([r['image'] for r in rows]),'target':torch.stack([r['target'] for r in rows]),
            'native_target':[r['native_target'] for r in rows],'sample_id':[r['sample_id'] for r in rows]}

def seed_worker(_):
    import cv2,torch
    cv2.setNumThreads(1);torch.set_num_threads(1)
