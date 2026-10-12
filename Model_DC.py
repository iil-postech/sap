"""
Official implementation of "Spectral-Aligned Pruning for Universal Error-Correcting Code Transformers" (SAP)
Information and Intelligence Lab (IIL)
Department of Electrical Engineering
Pohang University of Science and Technology (POSTECH), South Korea.
@author: Cho Sanghyeon, chosh320@postech.ac.kr

Built on FECCT, reimplemented by Cho Sanghyeon as the base decoder.
"""
from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


@torch.no_grad()
def prune_linear_layer(layer, index, dim=0):
    """Physically select output rows or input columns without an external helper."""
    index = index.to(layer.weight.device)
    result = copy.deepcopy(layer)
    result.weight = nn.Parameter(layer.weight.index_select(dim, index).clone(),
                                 requires_grad=layer.weight.requires_grad)
    if dim == 0:
        result.out_features = index.numel()
        if layer.bias is not None:
            result.bias = nn.Parameter(layer.bias.index_select(0, index).clone(),
                                       requires_grad=layer.bias.requires_grad)
    else:
        result.in_features = index.numel()
    return result


def sign_to_bin(x):
    return 0.5 * (1 - x)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if N > 1:
            self.norm2 = LayerNorm(layer.size)

    def forward(self, x, mask, head_masks=None):
        for idx, layer in enumerate(self.layers):
            layer_head_mask = head_masks[idx] if head_masks is not None else None
            x = layer(x, mask, head_mask=layer_head_mask)

            if (idx + 1) == len(self.layers)//2 and len(self.layers) > 1:
                x = self.norm2(x)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, head_mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, head_mask=head_mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, args, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.args = args
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.pruned_heads = set()
        d_hidden = 50
        self.one_d_mapping = nn.Sequential(nn.Linear(1, d_hidden), nn.ReLU(),nn.Linear(d_hidden, 1))


    def get_mask_from_pc_matrix(self, pc_matrix):
        mask_nk_nk = pc_matrix@pc_matrix.T
        mask_n_n = pc_matrix.T@pc_matrix
        tmp1 = torch.cat([mask_n_n,pc_matrix.T],1)
        tmp2 = torch.cat([pc_matrix,mask_nk_nk],1)
        return self.one_d_mapping(torch.cat([tmp1,tmp2],0).unsqueeze(-1)).squeeze().unsqueeze(0).unsqueeze(0)

    def forward(self, query, key, value, mask=None, head_mask=None):
        nbatches = query.size(0)
        current_h = self.linears[0].weight.shape[0] // self.d_k
        query, key, value = \
            [l(x).view(nbatches, -1, current_h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask)
        if head_mask is not None:
            x = x * head_mask.view(1, -1, 1, 1)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, current_h * self.d_k)
        return self.linears[-1](x)

    def prune_heads(self, heads):
        """Remove heads using their original, unpruned indices."""
        new_heads = set(heads) - self.pruned_heads
        if not new_heads:
            return
        original_heads = self.h + len(self.pruned_heads)
        available = [i for i in range(original_heads) if i not in self.pruned_heads]
        if not new_heads.issubset(available) or len(new_heads) >= self.h:
            raise ValueError('Invalid attention head mask')
        keep = torch.tensor([i for i, original in enumerate(available) if original not in new_heads],
                            device=self.linears[0].weight.device)
        index = (keep[:, None] * self.d_k + torch.arange(self.d_k, device=keep.device)).flatten()

        self.linears[0] = prune_linear_layer(self.linears[0], index)
        self.linears[1] = prune_linear_layer(self.linears[1], index)
        self.linears[2] = prune_linear_layer(self.linears[2], index)
        self.linears[3] = prune_linear_layer(self.linears[3], index, dim=1)

        self.h -= len(new_heads)
        self.pruned_heads.update(new_heads)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores*self.get_mask_from_pc_matrix(mask)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), None

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))



class DC_ECC_Transformer(nn.Module):
    def __init__(self, args):
        super(DC_ECC_Transformer, self).__init__()
        self.args = args
        c = copy.deepcopy
        attn = MultiHeadedAttention(args.h, args.d_model, args, dropout=args.dropout_attn)
        ff = PositionwiseFeedForward(args.d_model, args.d_model*4, args.dropout)
        self.src_embed_synd = torch.nn.Embedding(5, args.d_model)
        self.src_embed_magn = torch.nn.Parameter(torch.empty(
            (1, args.d_model)))
        self.decoder = Encoder(EncoderLayer(
            args.d_model, c(attn), c(ff), args.dropout), args.N_dec)
        self.oned_final_embed = torch.nn.Sequential(
            *[nn.Linear(args.d_model, 1)])
        self.synd_to_mag = nn.Linear(args.d_model, args.d_model)
        self.mag_to_mag = nn.Linear(args.d_model, args.d_model)

        for name, p in self.named_parameters():
            if p.dim() > 1 and 'mask_emb' not in name and 'src_embed_synd' not in name:
                nn.init.xavier_uniform_(p)


    def forward(self, magnitude, syndrome, mask, z_mul, y, dict_batch, head_masks=None):
        emb_magn = self.src_embed_magn.unsqueeze(0) * magnitude.unsqueeze(-1)
        emb_synd = self.src_embed_synd(sign_to_bin(syndrome).long())
        mask = mask.to(torch.float32)
        dict_batch = dict_batch.to(torch.float32)
        emb = torch.cat([emb_magn, emb_synd], 1)
        emb = self.decoder(emb, mask, head_masks=head_masks)
        z_pred, loss, x_pred = self.loss(-emb, z_mul, y, dict_batch)
        return z_pred, loss.unsqueeze(-1), x_pred

    def loss(self, z_pred, z2, y, pc_matrix):
        n_max = z2.size(1)
        tmp = z_pred[:,n_max:].unsqueeze(-1)*pc_matrix.unsqueeze(0).unsqueeze(2)
        z_pred = self.mag_to_mag(z_pred[:,:n_max])+self.synd_to_mag(tmp.permute(0,1,3,2)).sum(1)
        z_pred = self.oned_final_embed(z_pred).squeeze(-1)
        z_pred = z_pred[:,:z2.size(1)]

        loss = (F.binary_cross_entropy_with_logits(z_pred, sign_to_bin(torch.sign(z2)),reduction='none')).mean(-1).mean()
        x_pred = sign_to_bin(torch.sign(-z_pred * torch.sign(y)))
        return z_pred, loss, x_pred
