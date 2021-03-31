""" Define the sublayers in encoder/decoder layer """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from .modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, len_q, n_head, d_model, d_k, d_v, dropout=0.1, relative=True):
        super(MultiHeadAttention, self).__init__()

        self.relative = relative

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        #nn.init.kaiming_uniform_(self.w_qs.weight)
        #nn.init.kaiming_uniform_(self.w_ks.weight)
        #nn.init.kaiming_uniform_(self.w_vs.weight)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.zeros_(self.w_qs.bias)
        nn.init.zeros_(self.w_ks.bias)
        nn.init.zeros_(self.w_vs.bias)
        if self.relative:
            self.w_es = Parameter(torch.FloatTensor(n_head, len_q, d_k))
            nn.init.normal_(self.w_es, mean=0, std=np.sqrt(2.0 / (len_q + d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        #nn.init.kaiming_uniform_(self.fc.weight)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        #assert(len_q == len_k == len_v)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(0, 2, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(0, 2, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(0, 2, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if self.relative:
            assert(len_q == len_k == len_v)
            e = self.w_es[:, -len_q:, :].unsqueeze(0).repeat(sz_b, 1, 1, 1).view(-1, len_q, d_k)  # (n*b) x lq x dk
        else:
            e = None

        mask = mask.unsqueeze(1).repeat(1, n_head, 1, 1).view(-1, len_q, len_k)
        #mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, e, mask=mask)

        output = output.view(sz_b, n_head, len_q, d_v)
        output = output.permute(0, 2, 1, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        nn.init.kaiming_uniform_(self.w_1.weight)
        nn.init.zeros_(self.w_1.bias)
        self.w_2 = nn.Linear(d_hid, d_in)
        nn.init.kaiming_uniform_(self.w_2.weight)
        nn.init.zeros_(self.w_2.bias)
        #self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        #self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x
        #output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        #output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
