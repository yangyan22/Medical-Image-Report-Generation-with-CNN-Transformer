import torch
import torch.nn as nn
from modules.encoder_decoder import EncoderDecoder
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, args, tokenizer):
        super(Model, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained

        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = nn.ModuleList(model.children())[:-2]
        self.resnet_f = nn.Sequential(*modules)

        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = nn.ModuleList(model.children())[:-2]
        self.resnet_l = nn.Sequential(*modules)

        self.avg_fnt = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

    def forward(self, images, targets=None, mode='train'):
        patch_feats = self.resnet_f(images[:, 0])

        avg_feats_0 = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))  # [16, 512]
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats_0 = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)  # [16, 49, 512]
        
        patch_feats = self.resnet_l(images[:, 1])
        avg_feats_1 = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))  # [16, 512]
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats_1 = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)  # [16, 49, 512]

        fc_feats = torch.cat((avg_feats_0, avg_feats_1), dim=1) 
        att_feats = torch.cat((patch_feats_0, patch_feats_1), dim=1) 
        if mode == 'train':
            output = self.encoder_decoder(att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output
