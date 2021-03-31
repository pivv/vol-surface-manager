from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, e=None, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))  # (n*b) x lq x lk
        if e is not None:
            s = torch.bmm(q, e.transpose(1, 2))  # (n*b) x lq x lq
            sz_b, len_q, _ = s.size()
            s = F.pad(s, (1, 0)).view(-1, len_q+1, len_q)[:, 1:, :]  # (n*b) x lq x lq
            attn = attn + s

        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        if mask is not None:
            attn = attn.masked_fill(mask, 0.)

        attn = self.dropout(attn)
        output = torch.bmm(attn, v)  # (n*b) x lq x dv

        return output, attn
