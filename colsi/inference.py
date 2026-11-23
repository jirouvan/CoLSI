"""RGB -> visible mask -> frozen MoGe/RGB descriptors -> MLP tilt."""
from pathlib import Path
import numpy as np
from PIL import Image
from .runtime import configure_cuda,device_report,weight_path,write_json

class Pipeline:
    def __init__(self,weights_dir,*,segmentation_only=False):
        import torch
        from .model import build,load_weights
        self.device=configure_cuda();self.weights_dir=Path(weights_dir)
        self.segmentation=build();load_weights(self.segmentation,weight_path(weights_dir,'colsi_net.pt'))
        self.segmentation.to(self.device).eval().requires_grad_(False)
        self.moge=self.rgb=self.head=None
        if not segmentation_only:
            from .geometry import load_moge
            from .angle import load_angle_models
            self.moge=load_moge(weight_path(weights_dir,'moge3_vitl.pt'),self.device)
            self.rgb,self.head=load_angle_models(weights_dir,self.device)
        self.audit=device_report(self.segmentation)

    def segment(self,images):
        import torch
        from .model import predict
        from .segmentation import prepare_rgb,restore
        x=torch.stack([prepare_rgb(v,(288,512))[0] for v in images]).to(self.device)
        if any(p.device!=x.device for p in self.segmentation.parameters()):raise RuntimeError('Segmentation device mismatch')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):score=predict(self.segmentation,x)
        values=score.float().cpu().numpy();self.audit=device_report(self.segmentation,x)
        result=[]
        for rgb,logit in zip(images,values):
            label,prob=restore(logit,rgb.shape[:2],(288,512))
            result.append((label,prob))
        return result

    def objects(self,rgb,K,gravity,label,probability,sample_id):
        import torch
        from .task import proposals,landing_status
        from .geometry import predict_geometry
        from .geometry_features import extract_descriptor
        from .crops import crop_with_intrinsics,normalize_crop
        from .angle import extract_rgb_features,make_fusion_matrix
        if probability is None:return []
        objects,_=proposals({'semantic':label,'foreground_probability':1-probability[0],
                            'class_confidence':probability.max(0)},sample_id)
        if not objects:return []
        if self.moge is None:return [{**o,'surface_tilt_deg':None,'landing_status':None} for o in objects]
        fields=predict_geometry(self.moge,rgb,K,self.device)
        crops=[];aux=[];geometry=[];valid=[]
        for obj in objects:
            crop=crop_with_intrinsics(rgb,obj['mask'],K,gravity,input_size=224,margin_ratio=.15,appearance_mode='bbox')
            desc=extract_descriptor(fields['axial_z'],fields['normal_direct'],fields['valid_depth'],fields['valid_normal'],obj['mask'],K,gravity)
            crops.append(normalize_crop(crop['crop']));aux.append(crop['aux']);geometry.append(desc['values']);valid.append(crop['valid'])
        x=torch.stack(crops).to(self.device).float();q=torch.as_tensor(np.asarray(aux,np.float32),device=self.device)
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            value=extract_rgb_features(self.rgb,x,q)
            matrix=make_fusion_matrix(value['embedding'].cpu().numpy(),value['aux_normalized'].cpu().numpy(),value['rgb_angle_deg'].cpu().numpy(),np.asarray(geometry,np.float32))
            result=self.head.predict(torch.as_tensor(matrix,device=self.device))
        angle=result['angle_deg'].cpu().numpy();ok=result['valid'].cpu().numpy()&np.asarray(valid,bool)
        for i,obj in enumerate(objects):
            tilt=float(angle[i]) if ok[i] else None
            obj.update(surface_tilt_deg=tilt,landing_status=landing_status(tilt),angle_deg=tilt,angle_valid=bool(ok[i]))
        return objects

def save_frame(output,sample_id,label,objects):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    Image.fromarray(label).save(output/'semantic.png')
    serial=[]
    for i,obj in enumerate(objects):
        row={k:v for k,v in obj.items() if k!='mask'}
        path=output/f'object_{i:04d}.png';Image.fromarray(obj['mask'].astype(np.uint8)*255).save(path)
        row['visible_mask']=path.name;serial.append(row)
    write_json(output/'prediction.json',{'sample_id':sample_id,'objects':serial})
    return serial
