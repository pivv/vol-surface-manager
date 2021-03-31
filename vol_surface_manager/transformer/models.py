""" Define the Transformer model """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from .layers import EncoderLayer, DecoderLayer, MultiDecoderLayer, PositionwiseLayer


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table) # n_position x d_hid


def get_pos_mask(pos, pad):
    return pos.ne(pad)


def flatten_maybe_padded_sequences(maybe_padded_seq, pos, pad):
    def flatten_unpadded_sequences():
        return maybe_padded_seq.view(
            (-1,) + maybe_padded_seq.size()[2:])
    if pos is None:
        return flatten_unpadded_sequences()

    def flatten_padded_sequences():
        pos_mask = get_pos_mask(pos, pad)
        return maybe_padded_seq.view((-1,) +
            maybe_padded_seq.size()[2:])[pos_mask.view((-1,))]

    return flatten_padded_sequences()


def get_pad_mask(seq, pos, pad, d_model):
    pos_mask = get_pos_mask(pos, pad).type(torch.bool)
    padding_mask = pos_mask.eq(0).type(torch.bool)
    padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, d_model)  # b x lq x d_model
    return padding_mask


def get_non_pad_mask(seq, pos, pad):
    pos_mask = get_pos_mask(pos, pad)
    return pos_mask.unsqueeze(-1).type(torch.float)


def get_attn_key_pad_mask(seq_k, pos_k, seq_q, pad):
    len_q = seq_q.size(1)
    pos_mask = get_pos_mask(pos_k, pad).type(torch.bool)
    padding_mask = pos_mask.eq(0).type(torch.bool)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """
    if len(seq.size()) == 2:
        sz_b, len_s = seq.size()
    else:
        assert(len(seq.size()) == 3)
        sz_b, len_s, _ = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1).type(torch.bool)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


class PositionwiseEncoder(nn.Module):
    def __init__(
            self,
            d_src, len_max, n_layers, d_model, d_inner,
            dropout=0.1, vector_input=False, position=True, sinusoid=True):

        super(PositionwiseEncoder, self).__init__()

        self.position = position
        self.pad = -1

        if vector_input:
            self.src_emb = nn.Linear(d_src, d_model, bias=False)
            nn.init.normal_(self.src_emb.weight, mean=0, std=np.sqrt(2.0 / (d_src + d_model)))
        else:
            self.src_emb = nn.Embedding(d_src, d_model)
            nn.init.normal_(self.src_emb.weight, mean=0, std=np.sqrt(2.0 / (d_src + d_model)))

        if self.position:
            if sinusoid:
                self.position_enc = nn.Embedding.from_pretrained(
                    get_sinusoid_encoding_table(len_max+1, d_model, padding_idx=0), freeze=True)
            else:
                self.position_enc = nn.Embedding(len_max+1, d_model, padding_idx=0)
                nn.init.normal_(self.position_enc.weight, mean=0, std=np.sqrt(2.0 / (len_max + d_model)))

        self.layer_stack = nn.ModuleList([
            PositionwiseLayer(d_model, d_inner, dropout=dropout)
            for _ in range(n_layers)])

        self.d_model = d_model

    def forward(self, src_seq, src_pos):
        # src_seq : sz_b x len_max x d_src

        # -- Prepare masks
        pad_mask = get_pad_mask(src_seq, src_pos, self.pad, self.d_model)

        # -- Forward
        if self.position:
            enc_output = self.src_emb(src_seq) + self.position_enc(src_pos + 1)  # sz_b x len_max x d_model
        else:
            enc_output = self.src_emb(src_seq)  # sz_b x len_max x d_model

        for pos_layer in self.layer_stack:
            enc_output = pos_layer(
                enc_output,
                pad_mask=pad_mask)

        return enc_output


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            d_src, len_max, n_layers, n_head, d_k, d_v, d_model, d_inner,
            dropout=0.1, vector_input=False, position=True, sinusoid=True, relative=True):

        super(Encoder, self).__init__()

        self.position = position
        self.pad = -1

        if vector_input:
            self.src_emb = nn.Linear(d_src, d_model)
            nn.init.normal_(self.src_emb.weight, mean=0, std=np.sqrt(2.0 / (d_src + d_model)))
        else:
            self.src_emb = nn.Embedding(d_src, d_model)
            nn.init.normal_(self.src_emb.weight, mean=0, std=np.sqrt(2.0 / (d_src + d_model)))

        if self.position:
            if sinusoid:
                self.position_enc = nn.Embedding.from_pretrained(
                    get_sinusoid_encoding_table(len_max+1, d_model, padding_idx=0), freeze=True)
            else:
                self.position_enc = nn.Embedding(len_max+1, d_model, padding_idx=0)
                nn.init.normal_(self.position_enc.weight, mean=0, std=np.sqrt(2.0 / (len_max + d_model)))

        self.layer_stack = nn.ModuleList([
            EncoderLayer(len_max, d_model, d_inner, n_head, d_k, d_v, dropout=dropout, relative=relative)
            for _ in range(n_layers)])

        self.d_model = d_model

    def forward(self, src_seq, src_pos, base_output=None, use_mask=True, return_attns=False):
        # src_seq : sz_b x len_max x d_src

        enc_slf_attn_list = []

        # -- Prepare masks
        pad_mask = get_pad_mask(src_seq, src_pos, self.pad, self.d_model)
        slf_attn_mask_subseq = get_subsequent_mask(src_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(src_seq, src_pos, src_seq, self.pad)
        if use_mask:
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask = slf_attn_mask_keypad

        # -- Forward
        if self.position:
            enc_output = self.src_emb(src_seq) + self.position_enc(src_pos + 1)  # sz_b x len_max x d_model
        else:
            enc_output = self.src_emb(src_seq)  # sz_b x len_max x d_model

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, base_output,
                pad_mask=pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class Decoder(nn.Module):
    """ A decoder model with self attention mechanism. """

    def __init__(
            self,
            d_tgt, len_max, n_layers, n_head, d_k, d_v, d_model, d_inner,
            dropout=0.1, vector_input=False, position=True, sinusoid=True, relative=True):

        super(Decoder, self).__init__()

        self.position = position
        self.pad = -1

        if vector_input:
            self.tgt_emb = nn.Linear(d_tgt, d_model)
            nn.init.normal_(self.tgt_emb.weight, mean=0, std=np.sqrt(2.0 / (d_tgt + d_model)))
        else:
            self.tgt_emb = nn.Embedding(d_tgt, d_model)
            nn.init.normal_(self.tgt_emb.weight, mean=0, std=np.sqrt(2.0 / (d_tgt + d_model)))

        if self.position:
            if sinusoid:
                self.position_enc = nn.Embedding.from_pretrained(
                    get_sinusoid_encoding_table(len_max+1, d_model, padding_idx=0), freeze=True)
            else:
                self.position_enc = nn.Embedding(len_max+1, d_model, padding_idx=0)
                nn.init.normal_(self.position_enc.weight, mean=0, std=np.sqrt(2.0 / (len_max + d_model)))

        self.layer_stack = nn.ModuleList([
            DecoderLayer(len_max, d_model, d_inner, n_head, d_k, d_v, dropout=dropout, relative=relative)
            for _ in range(n_layers)])

        self.d_model = d_model

    def forward(self, tgt_seq, tgt_pos, src_seq, src_pos, enc_output, base_output=None,
                use_mask=True, return_attns=False, init_dec_enc_attn_mask=None):
        # src_seq : sz_b x len_max x d_src

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        pad_mask = get_pad_mask(tgt_seq, tgt_pos, self.pad, self.d_model)
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(tgt_seq, tgt_pos, tgt_seq, self.pad)
        if use_mask:
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask = slf_attn_mask_keypad
        dec_enc_attn_mask = get_attn_key_pad_mask(src_seq, src_pos, tgt_seq, self.pad)  # b x lq x lk
        if init_dec_enc_attn_mask is not None:
            dec_enc_attn_mask = (dec_enc_attn_mask + init_dec_enc_attn_mask).gt(0)

        # -- Forward
        if self.position:
            dec_output = self.tgt_emb(tgt_seq) + self.position_enc(tgt_pos + 1)  # sz_b x len_max x d_model
        else:
            dec_output = self.tgt_emb(tgt_seq)  # sz_b x len_max x d_model

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, base_output,
                pad_mask=pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            d_src, len_max_src, d_tgt, len_max_tgt,
            n_layers_enc, n_head_enc, d_k_enc, d_v_enc, d_model_enc, d_inner_enc,
            dropout_enc, vector_input_enc, position_enc, sinusoid_enc, relative_enc,
            n_layers_dec, n_head_dec, d_k_dec, d_v_dec, d_model_dec, d_inner_dec,
            dropout_dec, vector_input_dec, position_dec, sinusoid_dec, relative_dec):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            d_src, len_max_src, n_layers_enc, n_head_enc, d_k_enc, d_v_enc, d_model_enc, d_inner_enc,
            dropout_enc, vector_input_enc, position_enc, sinusoid_enc, relative_enc)

        self.decoder = Decoder(
            d_tgt, len_max_tgt, n_layers_dec, n_head_dec, d_k_dec, d_v_dec, d_model_dec, d_inner_dec,
            dropout_dec, vector_input_dec, position_dec, sinusoid_dec, relative_dec)

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos, base_output_enc=None, base_output_dec=None,
                use_mask_enc=False, use_mask_dec=True, init_dec_enc_attn_mask=None):
        enc_output = self.encoder(src_seq, src_pos, base_output_enc, use_mask=use_mask_enc)
        dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, src_pos, enc_output, base_output_dec,
                                  use_mask=use_mask_dec, init_dec_enc_attn_mask=init_dec_enc_attn_mask)
        return dec_output


class MultiDecoder(nn.Module):
    """ A decoder model with self attention mechanism. """

    def __init__(
            self,
            n_encs, d_tgt, len_max, n_layers, n_head, d_k, d_v, d_model, d_inner,
            dropout=0.1, vector_input=False, position=True, sinusoid=True, relative=True):

        super(MultiDecoder, self).__init__()

        self.position = position
        self.pad = -1

        if vector_input:
            self.tgt_emb = nn.Linear(d_tgt, d_model)
            nn.init.normal_(self.tgt_emb.weight, mean=0, std=np.sqrt(2.0 / (d_tgt + d_model)))
        else:
            self.tgt_emb = nn.Embedding(d_tgt, d_model)
            nn.init.normal_(self.tgt_emb.weight, mean=0, std=np.sqrt(2.0 / (d_tgt + d_model)))

        if self.position:
            if sinusoid:
                self.position_enc = nn.Embedding.from_pretrained(
                    get_sinusoid_encoding_table(len_max+1, d_model, padding_idx=0), freeze=True)
            else:
                self.position_enc = nn.Embedding(len_max+1, d_model, padding_idx=0)
                nn.init.normal_(self.position_enc.weight, mean=0, std=np.sqrt(2.0 / (len_max + d_model)))

        self.layer_stack = nn.ModuleList([
            MultiDecoderLayer(n_encs, len_max, d_model, d_inner, n_head, d_k, d_v, dropout=dropout, relative=relative)
            for _ in range(n_layers)])

        self.d_model = d_model

    def forward(self, tgt_seq, tgt_pos, srcs_seq, srcs_pos, encs_output, base_output=None,
                use_mask=True, return_attns=False, init_dec_encs_attn_mask=None):
        # src_seq : sz_b x len_max x d_src

        dec_slf_attn_list, dec_encs_attn_list = [], []

        # -- Prepare masks
        pad_mask = get_pad_mask(tgt_seq, tgt_pos, self.pad, self.d_model)
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(tgt_seq, tgt_pos, tgt_seq, self.pad)
        if use_mask:
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask = slf_attn_mask_keypad

        dec_encs_attn_mask = []
        for (index_enc, src_seq), src_pos, enc_output in zip(enumerate(srcs_seq), srcs_pos, encs_output):
            init_dec_enc_attn_mask = None
            if init_dec_encs_attn_mask is not None:
                init_dec_enc_attn_mask = init_dec_encs_attn_mask[index_enc]
            dec_enc_attn_mask = get_attn_key_pad_mask(src_seq, src_pos, tgt_seq, self.pad)  # b x lq x lk
            if init_dec_enc_attn_mask is not None:
                dec_enc_attn_mask = (dec_enc_attn_mask + init_dec_enc_attn_mask).gt(0)
            dec_encs_attn_mask.append(dec_enc_attn_mask)

        # -- Forward
        if self.position:
            dec_output = self.tgt_emb(tgt_seq) + self.position_enc(tgt_pos + 1)  # sz_b x len_max x d_model
        else:
            dec_output = self.tgt_emb(tgt_seq)  # sz_b x len_max x d_model

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_encs_attn = dec_layer(
                dec_output, encs_output, base_output,
                pad_mask=pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_encs_attn_mask=dec_encs_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_encs_attn_list += [dec_encs_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_encs_attn_list
        return dec_output


class MultiTransformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            n_encs, d_srcs, len_max_srcs, d_tgt, len_max_tgt,
            n_layers_encs, n_head_encs, d_k_encs, d_v_encs, d_model_encs, d_inner_encs,
            dropout_encs, vector_input_encs, position_encs, sinusoid_encs, relative_encs,
            n_layers_dec, n_head_dec, d_k_dec, d_v_dec, d_model_dec, d_inner_dec,
            dropout_dec, vector_input_dec, position_dec, sinusoid_dec, relative_dec):

        super(MultiTransformer, self).__init__()

        self.encoders = nn.ModuleList([
            Encoder(
                d_src, len_max_src, n_layers_enc, n_head_enc, d_k_enc, d_v_enc, d_model_enc, d_inner_enc,
                dropout_enc, vector_input_enc, position_enc, sinusoid_enc, relative_enc)
            for d_src, len_max_src, n_layers_enc, n_head_enc, d_k_enc, d_v_enc, d_model_enc, d_inner_enc,
                dropout_enc, vector_input_enc, position_enc, sinusoid_enc, relative_enc in zip(
                d_srcs, len_max_srcs, n_layers_encs, n_head_encs, d_k_encs, d_v_encs, d_model_encs, d_inner_encs,
                dropout_encs, vector_input_encs, position_encs, sinusoid_encs, relative_encs)])

        self.decoder = MultiDecoder(
            n_encs, d_tgt, len_max_tgt, n_layers_dec, n_head_dec, d_k_dec, d_v_dec, d_model_dec, d_inner_dec,
            dropout_dec, vector_input_dec, position_dec, sinusoid_dec, relative_dec)

    def forward(self, srcs_seq, srcs_pos, tgt_seq, tgt_pos, base_output_enc=None, base_output_dec=None,
                use_mask_encs=None, use_mask_dec=True, init_dec_encs_attn_mask=None):
        encs_output = []
        if use_mask_encs is None:
            use_mask_encs = [False] * len(self.encoders)
        for encoder, src_seq, src_pos, use_mask_enc in zip(self.encoders, srcs_seq, srcs_pos, use_mask_encs):
            enc_output = encoder(src_seq, src_pos, base_output_enc, use_mask=use_mask_enc)
            encs_output.append(enc_output)
        dec_output = self.decoder(tgt_seq, tgt_pos, srcs_seq, srcs_pos, encs_output, base_output_dec,
                                  use_mask=use_mask_dec, init_dec_encs_attn_mask=init_dec_encs_attn_mask)
        return dec_output
