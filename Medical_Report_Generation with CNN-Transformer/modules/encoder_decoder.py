from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .att_model import AttModel


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):  # q, k, v 16, 8, 98, 64
    d_k = query.size(-1)  # 64
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # BS, h, L, d/h  * BS, h, d/h, L  # [16, 8, L, L]
   
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)   # [16, h, L, L]  
    p_attn = F.softmax(scores, dim=-1)    # [16, h, L, L]  

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn 


def subsequent_mask(size):  
    attn_shape = (1, size, size)  # (1, 59, 59)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')  
    return torch.from_numpy(subsequent_mask) == 0 


class Encoder(nn.Module):
    def __init__(self, layer, N): 
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model) 

    def forward(self, x, mask):  
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module): 
    def __init__(self, d_model, self_attn, feed_forward, dropout):  # 512 MHA PFF 0.1
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):  
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  
        return self.sublayer[1](x, self.feed_forward) 


class SublayerConnection(nn.Module): 
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # print(sublayer) 
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):  # Norm
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class PositionwiseFeedForward(nn.Module):  
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # [16, 98, 512]
        return self.w_2(self.dropout(F.relu(self.w_1(x))))  


class MultiHeadedAttention(nn.Module): 
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  
        self.d_k = d_model // h 
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):  # [BS, 98, 512]  [BS, L, d_model] 
        if mask is not None:
            mask = mask.unsqueeze(1)  # [BS, 1, 1, L]
        nbatches = query.size(0)
       
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  
             for l, x in zip(self.linears, (query, key, value))]  # [BS, L, d_model] ->[BS, h, L, d_model/h]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)  # [16, 8, 98, 64]
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)  #  [16, 98, 512]
        return self.linears[-1](x)  # fc [16, 98, 512]


class PositionalEncoding(nn.Module):  # PE
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # [5000, 512]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # 0-4999 [5000, 1]
        # print(torch.arange(0, d_model, 2))  # 0-510
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *  
                             -(math.log(10000.0) / d_model))  # -0.017988946039015984  
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  # [5000, 512] 

        pe = pe.unsqueeze(0)  # torch.Size([1, 5000, 512])
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]  # [16, 59, 512]  
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.lut = nn.Embedding(vocab, d_model)  # Embedding(len(vocab), 512) if not pretrained
        # glove_file = datapath('/home/camlab1/R2Gen/Y_data/word_vector/mimc/pro_512/vectors.txt')  # If pretrained
        # tmp_file = get_tmpfile("./test_word2vec.txt")
        # glove2word2vec(glove_file, tmp_file)
        # wvmodel = KeyedVectors.load_word2vec_format(tmp_file)
        # weight = torch.zeros(3974, 512)
        # for i in range(3974):
        #     weight[i, :] = torch.from_numpy(wvmodel.get_vector(str(i)))
        # self.lut = nn.Embedding.from_pretrained(weight)
        # print("pretrained word embedding loaded")

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Decoder(nn.Module):
    def __init__(self, layer, N):  # layer
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):  # self.d_model, MHA1, MHA2, FFN, self.dropout)
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):  # tgt, memory: encoder feature, src mask, tgt mask 
        # print('tgt')
        # print(x.shape)   # [2, 44, 512]
        # print('tgt_mask')   # [2, 44, 44]  True False 
        # print(tgt_mask.shape)
        # print('memory')
        # print(memory.shape)  # [2, 98, 512]
        # print('src_mask')
        # print(src_mask.shape)   # [2, 1, 98]  
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))  
        return self.sublayer[2](x, self.feed_forward)  # [2, 44, 512]


class Transformer(nn.Module):   
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  
        self.tgt_embed = tgt_embed  

    def forward(self, src, tgt, src_mask, tgt_mask):  # att_feats, seq, att_masks, seq_mask [16, L, 512]  [16, 59]  [16, 1, L]  [16, 59, 59]
        return self.decode(tgt, self.encode(src, src_mask), src_mask, tgt_mask) 
    def encode(self, src, src_mask):   # [8, 98, 512] [8, 1, 98]
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class EncoderDecoder(AttModel):  
    def __init__(self, args, tokenizer):  
        super(EncoderDecoder, self).__init__(args, tokenizer)  
        self.args = args
        self.num_layers = args.num_layers  # 3
        self.d_model = args.d_model  # 512
        self.d_ff = args.d_ff  # 512
        self.num_heads = args.num_heads  # 8
        self.dropout = args.dropout  # 0.1
        tgt_vocab = self.vocab_size + 1  
        self.model = self.make_model(tgt_vocab)  
        self.logit = nn.Linear(args.d_model, tgt_vocab)  # 512->738 


    def make_model(self, tgt_vocab):  
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        # encoder, decoder, src_embed, tgt_embed
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),  
            Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout), self.num_layers),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)))
        # randomly initiate
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def _prepare_feature_forward(self, att_feats, seq=None):  # [16, L_V_feature, 2048])
        # Clip the length of att_masks and att_feats to the maximum length
        # att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = self.att_embed(att_feats)  # self.att_embed(att_feats) [2, 98, 512]
        att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)  # [bs, 98]
        att_masks = att_masks.unsqueeze(-2)  #  [bs, 1, 98]

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True
            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None
        return att_feats, seq, att_masks, seq_mask  # [16, 98, 512]  [16, 59]  [16, 1, 98]  [16, 59, 59]

    # fc_feats[16, 1024], att_feats[16, 98, 512], targets[16, 60], {'mode': 'forward'}
    def _forward(self,  att_feats, seq):  # batch[8, 98, 2048] target[16, 60]
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, seq)  # [16, 98, 512] [16, 59] [16, 1, 98] [16, 59, 59]
        out = self.model(att_feats, seq, att_masks, seq_mask)  # [2, 44, 512]
        outputs = F.log_softmax(self.logit(out), dim=-1)  # [2, 44, 761]
        return outputs

