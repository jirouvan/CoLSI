"""Canonical B8 segmentation and prediction-first full-chain evaluation."""
from pathlib import Path
import numpy as np
from PIL import Image
from .runtime import write_json
from .inference import Pipeline,save_frame
from .data import records,read_image

def evaluate(data_root,weights_dir,output,*,split='val',task='segmentation'):
    from .segmentation import confusion,boundary_counts,metrics
    from .benchmark import evaluate_predictions
    from .task import aggregate
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Select a fresh evaluation output directory')
    out.mkdir(parents=True,exist_ok=True)
    public=records(data_root,split,supervision=False);sim=Path(data_root)/'CoLSI-Sim'
    pipeline=Pipeline(weights_dir,segmentation_only=task=='segmentation')
    for begin in range(0,len(public),8):
        group=public[begin:begin+8]
        images=[read_image(sim,r['rgb_path'],r['rgb_sha256'],rgb=True) for r in group]
        predicted=pipeline.segment(images)
        for row,rgb,(label,prob) in zip(group,images,predicted):
            objects=pipeline.objects(rgb,row['K_input'],row['g_input_C'],label,prob,row['sample_id']) if task=='full-chain' else []
            save_frame(out/'predictions'/row['sample_id'],row['sample_id'],label,objects)
        write_json(out/'progress.json',{'images':min(begin+8,len(public)),'total':len(public),'stage':'prediction'})
    write_json(out/'prediction_seal.json',{'status':'COMPLETE_PREDICTIONS','task':task,'split':split,
       'sample_ids':[r['sample_id'] for r in public],'segmentation_batch':8,'NMF_seed':61107,'segmentation_precision':'BF16',
       'RGB_MLP_precision':'FP32','RGB_batch_scope':'per-frame objects','TF32':False})
    truth=records(data_root,split,supervision=True)
    cm=np.zeros((8,9),np.int64);bc=np.zeros((7,4),np.int64);reports=[]
    for row in truth:
        sid=row['sample_id'];folder=out/'predictions'/sid
        label=np.asarray(Image.open(folder/'semantic.png'))
        target=read_image(sim,row['semantic_mask']['path'],row['semantic_mask']['sha256'])
        cm+=confusion(target,label);bc+=boundary_counts(target,label)
        if task=='full-chain':
            import json
            gt=[]
            for obj in row['objects']:
                mask=read_image(sim,obj['visible_mask']['path'],obj['visible_mask']['sha256'])>0
                gt.append({'object_id':sid+':instance:'+str(obj['instance_id']),'class_id':obj['class_id'],
                           'mask':mask,'angle_deg':obj['alpha_deg'],'angle_valid':True})
            pred=json.loads((folder/'prediction.json').read_text('utf8'))['objects']
            for obj in pred:obj['mask']=np.asarray(Image.open(folder/obj['visible_mask']))>0
            reports.append(evaluate_predictions([{'sample_id':sid,'objects':gt}],[{'sample_id':sid,'objects':pred}],
                iou_threshold=.5,angle_thresholds=(1,2,3,5,10),confidence_threshold=None))
    result={'split':split,'seed':17,'images':len(truth),'segmentation':metrics(cm,bc),'device':pipeline.audit}
    if reports:result['complete_task']=aggregate(reports)
    write_json(out/'metrics.json',result)
    return result
