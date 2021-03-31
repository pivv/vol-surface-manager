""" Define the Layers """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from .sublayers import MultiHeadAttention, PositionwiseFeedForward


class PositionwiseLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, dropout=0.1):
        super(PositionwiseLayer, self).__init__()
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, pad_mask=None):
        enc_output = self.pos_ffn(enc_input)
        enc_output = enc_output.masked_fill(pad_mask, 0.)

        return enc_output


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, len_q, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, relative=True):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            len_q, n_head, d_model, d_k, d_v, dropout=dropout, relative=relative)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, base_output=None, pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = enc_output.masked_fill(pad_mask, 0.)

        if base_output is not None:
            enc_output = enc_output + base_output
            enc_output = enc_output.masked_fill(pad_mask, 0.)

        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.masked_fill(pad_mask, 0.)

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """ Compose with three layers """

    def __init__(self, len_q, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, relative=True):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(len_q, n_head, d_model, d_k, d_v, dropout=dropout, relative=relative)
        self.enc_attn = MultiHeadAttention(len_q, n_head, d_model, d_k, d_v, dropout=dropout, relative=False)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, base_output=None,
        pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output = dec_output.masked_fill(pad_mask, 0.)

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = dec_output.masked_fill(pad_mask, 0.)

        if base_output is not None:
            dec_output = dec_output + base_output
            dec_output = dec_output.masked_fill(pad_mask, 0.)

        dec_output = self.pos_ffn(dec_output)
        dec_output = dec_output.masked_fill(pad_mask, 0.)

        return dec_output, dec_slf_attn, dec_enc_attn


class MultiDecoderLayer(nn.Module):
    """ Compose with four layers """

    def __init__(self, n_encs, len_q, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, relative=True):
        super(MultiDecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(len_q, n_head, d_model, d_k, d_v, dropout=dropout, relative=relative)
        self.encs_attn = nn.ModuleList([
            MultiHeadAttention(len_q, n_head, d_model, d_k, d_v, dropout=dropout, relative=False)
            for _ in range(n_encs)])
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, encs_output, base_output=None,
        pad_mask=None, slf_attn_mask=None, dec_encs_attn_mask=None):
        assert(len(self.encs_attn) == len(encs_output))

        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output = dec_output.masked_fill(pad_mask, 0.)

        dec_encs_attn = []
        for (index_enc, enc_attn), enc_output in zip(enumerate(self.encs_attn), encs_output):
            dec_enc_attn_mask = None
            if dec_encs_attn_mask is not None:
                dec_enc_attn_mask = dec_encs_attn_mask[index_enc]
            dec_output, dec_enc_attn = enc_attn(
                dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
            dec_output = dec_output.masked_fill(pad_mask, 0.)
            dec_encs_attn.append(dec_enc_attn)

        if base_output is not None:
            dec_output = dec_output + base_output
            dec_output = dec_output.masked_fill(pad_mask, 0.)

        dec_output = self.pos_ffn(dec_output)
        dec_output = dec_output.masked_fill(pad_mask, 0.)

        return dec_output, dec_slf_attn, dec_encs_attn
