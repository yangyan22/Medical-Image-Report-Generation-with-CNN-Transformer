from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn


class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()  # self = EncoderDecoder()  nn.module

    def forward(self, *args, **kwargs):
        # fc_feats[16, 1024], att_feats[16, 98, 512], targets[16, 60], {'mode': 'forward'}
        
        mode = kwargs.get('mode', 'forward')  # mode = forward or sample
        if 'mode' in kwargs:
            del kwargs['mode']
        # EncoderDecoder._forward of EncoderDecoder or AttModel._sample of EncoderDecoder(
        return getattr(self, '_' + mode)(*args, **kwargs)  # [16, 59, 761]   or seq, seqLogprobs
