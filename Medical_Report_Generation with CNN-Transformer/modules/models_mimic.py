import torch
import torch.nn as nn
from modules.encoder_decoder import EncoderDecoder
import torchvision.models as models
import os


class Model(nn.Module):
    def __init__(self, args, tokenizer):
        super(Model, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = nn.ModuleList(model.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.avg_fnt = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

    def forward(self, images, targets=None, mode='train'):  
        patch_feats = self.resnet(images) 
        fc_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))  
        batch_size, feat_size, _, _ = patch_feats.shape
        att_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)  
        
        if mode == 'train':
            output = self.encoder_decoder(att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output