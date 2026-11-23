"""Frozen RGB letterbox, native restoration and segmentation metrics."""
import cv2
import numpy as np
import torch
CLASS_IDS=tuple(range(8))
def mapping(native_hw, size):
    h,w=map(int,native_hw);th,tw=map(int,size)
    if min(h,w,th,tw)<1:
        raise ValueError("Positive native/network shapes required")
    scale=min(th/h,tw/w);nh,nw=max(1,min(th,round(h*scale))),max(1,min(tw,round(w*scale)))
    top,left=(th-nh)//2,(tw-nw)//2;sx,sy=nw/w,nh/h
    A=np.array([[sx,0,left+.5*sx-.5],[0,sy,top+.5*sy-.5],[0,0,1]],np.float64)
    return (nh,nw),(top,left),A

def prepare_rgb(rgb,size):
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise ValueError("Native uint8 RGB required")
    (nh,nw),(top,left),A=mapping(rgb.shape[:2],size)
    image=np.full((*size,3),114,np.uint8);image[top:top+nh,left:left+nw]=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_LINEAR)
    values=(image.astype(np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return torch.from_numpy(values.transpose(2,0,1).copy()),A

def prepare(rgb, semantic, size):
    if (semantic.dtype!=np.uint8 or semantic.shape!=rgb.shape[:2] or not np.isin(semantic,(*CLASS_IDS,255)).all()):
        raise ValueError("Native uint8 semantic0..7,ignore255 required")
    image,A=prepare_rgb(rgb,size);(nh,nw),(top,left),_=mapping(semantic.shape,size)
    target=np.full(size,255,np.uint8);target[top:top+nh,left:left+nw]=cv2.resize(semantic,(nw,nh),interpolation=cv2.INTER_NEAREST_EXACT)
    return image,torch.from_numpy(target.astype(np.int64)),A

def restore(logits,native_hw,network_hw):
    """Network FP32 logits -> CPU FP64 softmax -> FP32/crop/native bilinear."""
    if logits.ndim!=3 or logits.shape[0]!=8 or tuple(logits.shape[1:])!=tuple(network_hw):
        raise ValueError("Explicit eight-class network-grid logits required")
    if not np.isfinite(logits).all():
        return np.full(native_hw,255,np.uint8),None
    x=logits.astype(np.float64);x-=x.max(axis=0,keepdims=True);p=np.exp(x);p=(p/p.sum(0,keepdims=True)).astype(np.float32)
    (nh,nw),(top,left),_=mapping(native_hw,network_hw)
    p=p[:,top:top+nh,left:left+nw]
    p=np.stack([cv2.resize(v,(native_hw[1],native_hw[0]),interpolation=cv2.INTER_LINEAR)for v in p])
    p/=np.maximum(p.sum(0,keepdims=True),np.finfo(np.float32).tiny)
    return p.argmax(0).astype(np.uint8),p

def confusion(target,prediction):
    valid=target!=255;pred=np.where((prediction>=0)&(prediction<8),prediction,8)
    return np.bincount((target[valid].astype(np.int64)*9+pred[valid]).ravel(),minlength=72).reshape(8,9)

def boundary_counts(target,prediction,tolerance=2):
    if tolerance!=2:
        raise ValueError("Fixed two-native-pixel boundary tolerance")
    valid=target!=255;counts=np.zeros((7,4),np.int64)
    kernel=np.ones((3,3),np.uint8);near=np.ones((5,5),np.uint8)
    for i,c in enumerate(range(1,8)):
        a=((target==c)&valid).astype(np.uint8);b=((prediction==c)&valid).astype(np.uint8)
        a=(a-cv2.erode(a,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0))>0
        b=(b-cv2.erode(b,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0))>0
        counts[i]=[(b&(cv2.dilate(a.astype(np.uint8),near)>0)).sum(),b.sum(),
                   (a&(cv2.dilate(b.astype(np.uint8),near)>0)).sum(),a.sum()]
    return counts

def metrics(cm,boundary):
    cm=np.asarray(cm,np.int64);b=np.asarray(boundary,np.int64)
    if cm.shape!=(8,9) or b.shape!=(7,4) or (cm<0).any() or (b<0).any():
        raise ValueError("Native complete confusion8x9/boundary7x4 required")
    tp=np.diag(cm[:,:8]).astype(float);gt=cm.sum(1);pred=cm[:,:8].sum(0);union=gt+pred-tp
    def divide(a,d):return np.divide(a,d,out=np.zeros_like(np.asarray(a,dtype=float)),where=d>0)
    iou=divide(tp,union);dice=divide(2*tp,gt+pred);precision=divide(tp,pred);recall=divide(tp,gt)
    active=union>0;fg=active.copy();fg[0]=False
    bp=divide(b[:,0],b[:,1]);br=divide(b[:,2],b[:,3]);bf=divide(2*bp*br,bp+br);ba=(b[:,1]+b[:,3])>0
    return {"foreground_mIoU":float(iou[fg].mean())if fg.any()else None,
        "all8_mIoU":float(iou[active].mean())if active.any()else None,"class_IoU":[float(iou[i])if active[i]else None for i in range(8)],
        "foreground_macro_Dice":float(dice[fg].mean())if fg.any()else None,
        "foreground_macro_Precision":float(precision[fg].mean())if fg.any()else None,
        "foreground_macro_Recall":float(recall[fg].mean())if fg.any()else None,
        "pixel_accuracy":float(tp.sum()/cm.sum())if cm.sum()else None,
        "foreground_macro_boundary_F1_2px":float(bf[ba].mean())if ba.any()else None,
        "eligible_pixels":int(cm.sum()),"failed_pixels":int(cm[:,8].sum()),"confusion8x9":cm.tolist(),
        "boundary_counts7x4":b.tolist(),"scope":"full_native_visible_semantic; no decoded_instance_or_angle_score"}
