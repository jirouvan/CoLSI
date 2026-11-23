"""Single-seed CoLSI-Net training and fixed-B8 validation."""
from pathlib import Path
import math,time,json
import numpy as np
from .runtime import configure_cuda,device_report,weight_path,write_json

def train(data_root,weights_dir,output,*,config_path,resume=False):
    import torch
    from torch.utils.data import DataLoader
    from .model import build,parameter_groups,train_output,predict
    from .data import SegmentationDataset,collate,seed_worker
    from .segmentation import restore,confusion,boundary_counts,metrics
    cfg=json.loads(Path(config_path).read_text('utf8'));t=cfg['training']
    if t['seed']!=17:raise ValueError('The published recipe uses seed17')
    if (t['validation_batch_size']!=8 or t['size_hw']!=[288,512] or t['precision']!='BF16'
        or t['tf32'] is not False or t['optimizer']!='AdamW' or t['gradient_clip_norm']!=1.
        or t['scheduler']!='single_cosine_floor0.1' or t['augmentation']!='none'):
        raise ValueError('This entry uses the registered B8/288x512/BF16/AdamW/cosine/CE-boundary training protocol')
    if t['effective_batch_size']%t['batch_size']:raise ValueError('Physical batch must divide effective batch')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'completion.json').exists():raise FileExistsError('Completed run already exists; select a fresh output path')
    if any(out.iterdir()) and not resume:raise FileExistsError('Use a fresh output path or explicit --resume')
    device=configure_cuda(17,t['threads'])
    trainset=SegmentationDataset(data_root,'train');valset=SegmentationDataset(data_root,'val')
    model=build(weight_path(weights_dir,'mscan_b_imagenet.pth')).to(device)
    optimizer=torch.optim.AdamW(parameter_groups(model,t),lr=t['lr'],weight_decay=t['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e:.1+.9*(1+math.cos(math.pi*e/t['epochs']))/2)
    rng=np.random.default_rng(17);start_epoch=1;best=-math.inf;history=[];updates=0
    if resume:
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        if saved['config']!=cfg:raise ValueError('Resume recipe differs')
        model.load_state_dict(saved['model_state_dict'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);rng.bit_generator.state=saved['rng']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start_epoch=saved['epoch']+1;best=saved['best'];history=saved['history'];updates=saved['updates']
    write_json(out/'config.json',cfg);began=time.perf_counter()
    for epoch in range(start_epoch,t['epochs']+1):
        order=rng.permutation(len(trainset)).tolist()
        groups=[order[i:i+t['effective_batch_size']] for i in range(0,len(order),t['effective_batch_size'])]
        batches=[g[j:j+t['batch_size']] for g in groups for j in range(0,len(g),t['batch_size'])]
        loader=DataLoader(trainset,batch_sampler=batches,num_workers=t['workers'],pin_memory=True,
                          collate_fn=collate,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(17+epoch),
                          **({'prefetch_factor':2} if t['workers'] else {}))
        iterator=iter(loader);model.train();loss_sum=0.
        for group in groups:
            optimizer.zero_grad(set_to_none=True)
            for _ in range(math.ceil(len(group)/t['batch_size'])):
                batch=next(iterator);x=batch['image'].to(device,non_blocking=True);y=batch['target'].to(device,non_blocking=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=train_output(model,x,y)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                (loss*(len(x)/len(group))).backward();loss_sum+=float(loss.detach())*len(x)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();updates+=1
        model.eval();cm=np.zeros((8,9),np.int64);bc=np.zeros((7,4),np.int64)
        val=DataLoader(valset,batch_size=8,shuffle=False,num_workers=t['workers'],pin_memory=True,collate_fn=collate,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(17))
        with torch.inference_mode():
            for batch in val:
                x=batch['image'].to(device,non_blocking=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):scores=predict(model,x)
                for score,target in zip(scores.float().cpu().numpy(),batch['native_target']):
                    label,_=restore(score,target.shape,(288,512));cm+=confusion(target,label);bc+=boundary_counts(target,label)
        result=metrics(cm,bc);score=result['foreground_mIoU'];scheduler.step()
        history.append({'epoch':epoch,'train_loss':loss_sum/len(trainset),'validation':result,'updates':updates})
        if score>best:
            best=score;torch.save({'model_state_dict':model.state_dict(),'epoch':epoch},out/'best.pt')
        payload={'model_state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
           'epoch':epoch,'best':best,'history':history,'updates':updates,'rng':rng.bit_generator.state,
           'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'config':cfg}
        tmp=out/'last.partial.pt';torch.save(payload,tmp);tmp.replace(out/'last.pt')
        write_json(out/'history.json',history);write_json(out/'device.json',device_report(model,x))
        print(json.dumps({'epoch':epoch,'val_mIoU':score,'best':best,'seconds':time.perf_counter()-began}),flush=True)
    write_json(out/'completion.json',{'status':'completed','epochs':t['epochs'],'seed':17,'best_val_mIoU':best,'updates':updates,'seconds':time.perf_counter()-began})
