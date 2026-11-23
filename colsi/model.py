"""CoLSI-Net: one MSCAN-B/LightHam backbone with the final CFA/QSF CCSA."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from . import _fusion as c
from . import _support as own
from . import _segnext
from .ccsa import FusionBlock
EVALUATION_SEED=61107
LOSS='CoLSI_fusion8_semantic_CE_single_boundary_BCE0.1_v1'
class Model(nn.Module):

    def __init__(self, base, variant, fusion, *, anchor_path=None, anchor_metadata=None, activation_checkpoint=False):
        super().__init__()
        self.base, self.variant, self.fusion = (base, variant, fusion)
        self.anchor_path = anchor_path
        self.loss_mode, self.evaluation_seed = (LOSS, EVALUATION_SEED)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.initialization = {**base.initialization, 'owned_variant': variant, 'single_backbone_Ham': True, 'single_boundary_head': True, 'boundary_weight': 0.1, 'activation_checkpoint': self.activation_checkpoint, 'all_eight_full_train_val_only': True, 'anchor': anchor_metadata}

    @property
    def backbone(self):
        return self.base.backbone

    @property
    def decode_head(self):
        return self.base.decode_head

    def train(self, mode=True):
        super().train(mode)
        return self

    def score(self, images, size, *, evaluation, disable_fusion=False, **controls):
        parts = c.extract(self.base, images)
        parent = c.resize(parts.coarse, size, evaluation)
        if disable_fusion:
            return (parent, None, {'fusion_disabled': True, 'reference': 'SegNeXt-B'})
        kwargs = {'anchor_logits': None, 'evaluation': evaluation, **controls}
        if self.activation_checkpoint and self.training and torch.is_grad_enabled():
            if any((isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.fusion.modules())):
                raise RuntimeError('Checkpoint only new BN-free fusion branches')
            output = checkpoint(self.fusion, parts.levels, parts.context, use_reentrant=False, preserve_rng_state=True, **kwargs)
        else:
            output = self.fusion(parts.levels, parts.context, **kwargs)
        quality, semantic = (output.get('quality'), output.get('semantic'))
        if quality is None:
            quality = torch.zeros_like(parent[:, :1])
        else:
            quality = c.resize(quality, size, evaluation)
        semantic = None if semantic is None else c.resize(semantic, size, evaluation)
        score = c.combine(parent, quality, semantic)
        info = {**output.get('diagnostic', {}), 'variant': self.variant, 'boundary_weight': 0.1, 'boundary_head_count': 1, 'semantic_LSE_count': int(semantic is not None), 'anchor_frozen': self.anchor_path is not None}
        return (score, output['boundary'], info)

    def forward(self, images):
        return self.score(images, images.shape[-2:], evaluation=False)[0]

def build(encoder_checkpoint=None,*,small_fixture=False):
    if encoder_checkpoint is None:
        base=_segnext.construct(small_fixture=small_fixture)
        base.initialization={'encoder':'loaded later from explicit full-model checkpoint'}
    else:
        base=_segnext.build({'model_id':'segnext_b','pretrained':True,'encoder_checkpoint':str(encoder_checkpoint)})
    base.evaluation_seed=EVALUATION_SEED
    channels=(8,16,32,64) if small_fixture else (64,128,320,512)
    width,hidden=(16,8) if small_fixture else (128,64)
    with torch.random.fork_rng(devices=[],enabled=True):
        fusion=FusionBlock('f05',c1_channels=channels[0],context_channels=base.decode_head.channels,
                          level_channels=channels[1:],width=width,heads=4,flow_hidden=hidden)
    model=Model(base,'f05',fusion,activation_checkpoint=False)
    model.public_model_id='colsi_net'
    model.display_name='CoLSI-Net'
    return model

def predict(model,images):
    if model.training:raise ValueError('Prediction requires eval mode')
    devices=[] if images.device.type=='cpu' else [images.device.index or 0]
    with torch.random.fork_rng(devices=devices,enabled=True):
        torch.manual_seed(EVALUATION_SEED)
        return model.score(images,images.shape[-2:],evaluation=True)[0]

def train_output(model,images,target):
    score,boundary,_=model.score(images,target.shape[-2:],evaluation=False)
    ce=F.cross_entropy(score,target,ignore_index=255)
    ce=torch.where(target.ne(255).any(),ce,score.sum()*0.)
    return ce+.1*own.boundary_loss(boundary,target)

def parameter_groups(model,training):
    return [{'params':[p],'lr':training['lr']*(.1 if n.startswith('base.backbone.') else 1.),
             'weight_decay':0. if 'norm' in n or 'pos_block' in n else training['weight_decay']}
            for n,p in model.named_parameters() if p.requires_grad]

def load_weights(model,path):
    value=torch.load(path,map_location='cpu',weights_only=True)
    state=value.get('model_state_dict',value.get('model',value))
    model.load_state_dict(state,strict=True)
    return value
