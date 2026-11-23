"""MSCAN-B/LightHam author adapter with explicit NMF input device."""
from pathlib import Path
import argparse
import re
import sys
import types
import torch
from torch import nn
from torch.nn import functional as F
from . import _author as author
LOSS='SegNeXt_author_MSCANB_LightHamHead_native_semantic_CE_v1'
class Registry:
    def register_module(self,*args,**kwargs):return lambda c:c

class BaseModule(nn.Module):
    def __init__(self,init_cfg=None):super().__init__();self.init_cfg=init_cfg
    def init_weights(self):
        if self.init_cfg is not None:raise ValueError('Adapter loads explicit pinned state, never implicit init_cfg IO')

def build_norm_layer(config,channels):
    config=dict(config);kind=config.pop('type');requires=config.pop('requires_grad',True)
    if kind=='GN':name='gn';layer=nn.GroupNorm(config.pop('num_groups'),channels,**config)
    elif kind in ('BN','BN2d','SyncBN'):
        name='bn';layer=(nn.SyncBatchNorm if kind=='SyncBN'else nn.BatchNorm2d)(channels,**config)
    else:raise ValueError('Unsupported author norm '+kind)
    for p in layer.parameters():p.requires_grad=requires
    return name,layer

class ConvModule(nn.Module):
    def __init__(self,cin,cout,kernel_size,conv_cfg=None,norm_cfg=None,act_cfg='default',**kwargs):
        super().__init__()
        if conv_cfg is not None:raise ValueError('Only author default Conv2d supported')
        bias=kwargs.pop('bias','auto');bias=(norm_cfg is None)if bias=='auto'else bias
        self.conv=nn.Conv2d(cin,cout,kernel_size,bias=bias,**kwargs)
        nn.init.kaiming_normal_(self.conv.weight,a=0,mode='fan_out',nonlinearity='relu')
        if self.conv.bias is not None:nn.init.zeros_(self.conv.bias)
        self.norm_name=None
        if norm_cfg is not None:
            self.norm_name,norm=build_norm_layer(norm_cfg,cout);self.add_module(self.norm_name,norm)
        self.activate=nn.ReLU(inplace=True)if act_cfg is not None else nn.Identity()
        if isinstance(act_cfg,dict)and act_cfg.get('type','ReLU')!='ReLU':raise ValueError('Unexpected author activation')
    def forward(self,x):
        x=self.conv(x)
        if self.norm_name:x=getattr(self,self.norm_name)(x)
        return self.activate(x)

class BaseDecodeHead(BaseModule):
    def __init__(self,in_channels,channels,num_classes,in_index,input_transform,dropout_ratio=.1,
            norm_cfg=None,align_corners=False,act_cfg=None,conv_cfg=None,**kwargs):
        super().__init__();self.in_channels=in_channels;self.in_index=in_index;self.channels=channels
        self.norm_cfg=norm_cfg;self.align_corners=align_corners;self.conv_cfg=conv_cfg
        self.act_cfg={'type':'ReLU'}if act_cfg is None else act_cfg
        if input_transform!='multiple_select':raise ValueError('Author head transform changed')
        self.dropout=nn.Dropout2d(dropout_ratio)if dropout_ratio>0 else nn.Identity()
        self.conv_seg=nn.Conv2d(channels,num_classes,1);nn.init.normal_(self.conv_seg.weight,0,.01);nn.init.zeros_(self.conv_seg.bias)
    def _transform_inputs(self,inputs):return [inputs[i]for i in self.in_index]
    def cls_seg(self,x):return self.conv_seg(self.dropout(x))

def _init_constant(module,val,bias=0):
    if getattr(module,'weight',None)is not None:nn.init.constant_(module.weight,val)
    if getattr(module,'bias',None)is not None:nn.init.constant_(module.bias,bias)

def _init_normal(module,mean=0,std=1,bias=0):
    if getattr(module,'weight',None)is not None:nn.init.normal_(module.weight,mean,std)
    if getattr(module,'bias',None)is not None:nn.init.constant_(module.bias,bias)

def _init_trunc(module,std=.02,bias=0):
    if getattr(module,'weight',None)is not None:nn.init.trunc_normal_(module.weight,std=std)
    if getattr(module,'bias',None)is not None:nn.init.constant_(module.bias,bias)

def author_modules():
    from timm.layers import DropPath
    root=author.VENDOR/'SegNeXt/mmseg/models';base='_colsi_segnext_c87'
    author.package(base,root);author.package(base+'.backbones',root/'backbones');author.package(base+'.decode_heads',root/'decode_heads')
    def module(**values):
        value=types.ModuleType('comparison_scoped_author_compat');value.__dict__.update(values);return value
    shims={'mmcv':module(),'mmcv.cnn':module(ConvModule=ConvModule,build_norm_layer=build_norm_layer),
        'mmcv.runner':module(BaseModule=BaseModule),'mmcv.cnn.bricks':module(DropPath=DropPath),
        'mmcv.cnn.utils':module(),'mmcv.cnn.utils.weight_init':module(constant_init=_init_constant,normal_init=_init_normal,trunc_normal_init=_init_trunc),
        'mmseg':module(),'mmseg.models':module(),'mmseg.models.builder':module(BACKBONES=Registry(),HEADS=Registry()),
        'mmseg.ops':module(resize=F.interpolate),base+'.builder':module(HEADS=Registry()),
        base+'.decode_heads.decode_head':module(BaseDecodeHead=BaseDecodeHead)}
    with author.temporary_modules(shims):
        backbone=author.load(base+'.backbones.mscan',root/'backbones/mscan.py')
        head=author.load(base+'.decode_heads.ham_head',root/'decode_heads/ham_head.py')
    if not getattr(head.NMF2D,'_colsi_explicit_device',False):
        original=head.NMF2D
        class InputDeviceNMF(original):
            _colsi_explicit_device=True
            def forward(self,x,return_bases=False):
                self.input_device=x.device
                return super().forward(x,return_bases)
            def _build_bases(self,B,S,D,R,cuda=False):
                # Keep author's CPU RNG stream and transfer-before-normalize order.
                bases=torch.rand((B*S,D,R),device='cpu').to(self.input_device)
                return F.normalize(bases,dim=1)
        head.NMF2D=InputDeviceNMF
    return backbone,head

class Model(nn.Module):
    def __init__(self,backbone,head):super().__init__();self.backbone=backbone;self.decode_head=head
    def forward(self,x):return self.decode_head(self.backbone(x))

def construct(*,small_fixture=False):
    backbone,head=author_modules()
    embed=[8,16,32,64]if small_fixture else[64,128,320,512]
    network=backbone.MSCAN(embed_dims=embed,mlp_ratios=[8,8,4,4],depths=[1]*4 if small_fixture else[3,3,12,3],drop_path_rate=.1)
    network.init_weights();channels=32 if small_fixture else 512
    decode=head.LightHamHead(in_channels=embed[1:],in_index=[1,2,3],channels=channels,ham_channels=channels,
        num_classes=8,norm_cfg={'type':'GN','num_groups':8 if small_fixture else 32,'requires_grad':True},
        align_corners=False,ham_kwargs={'MD_R':4}if small_fixture else{})
    return Model(network,decode)

def build(options):
    if options['model_id']!='segnext_b' or options.get('pretrained')is not True:raise ValueError('Pinned MSCAN-B initialization required')
    model=construct()
    # This official classification checkpoint carries a plain training Namespace.
    # Allow exactly that inert container, never arbitrary unrestricted pickle.
    with torch.serialization.safe_globals([argparse.Namespace]):
        source=torch.load(options['encoder_checkpoint'],map_location='cpu',weights_only=True)
    state=source.get('state_dict',source.get('model',source));state={k.removeprefix('module.').removeprefix('backbone.'):v for k,v in state.items()}
    state={k:v for k,v in state.items()if not k.startswith(('head.','norm.','classifier.'))}
    # The official OpenMMLab ImageNet conversion flattens the DWConv wrapper name.
    # This is a one-to-one name map; strict loading checks every tensor and shape.
    renamed={k:re.sub(r'\.mlp\.dwconv\.(weight|bias)$',r'.mlp.dwconv.dwconv.\1',k)for k in state}
    state={renamed[k]:v for k,v in state.items()}
    result=model.backbone.load_state_dict(state,strict=False)
    if result.missing_keys or result.unexpected_keys:raise ValueError('MSCAN-B state mismatch: '+str(result))
    model.initialization={'encoder_checkpoint':options['encoder_checkpoint'],'author_backbone_head':'unchanged MSCAN-B/LightHamHead',
        'device_adapter':'CPU random NMF bases transferred explicitly to input.device before normalization','loss':LOSS,
        'official_classification_DWConv_wrapper_name_mapping':{k:v for k,v in renamed.items()if k!=v}}
    return model

def train_output(model,images,target):
    logits=model(images);score=F.interpolate(logits,size=target.shape[-2:],mode='bilinear',align_corners=False)
    return F.cross_entropy(score,target,ignore_index=255),{'loss_mode':LOSS},logits

def predict(model,images):return model(images)

def parameter_groups(model,t):
    groups=[]
    for name,p in model.named_parameters():
        groups.append({'params':[p],'lr':t['lr']*(1 if name.startswith('decode_head.')else .1),
            'weight_decay':0. if 'norm'in name or 'pos_block'in name else t['weight_decay']})
    return groups
