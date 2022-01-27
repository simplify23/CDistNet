import math
import torch
import copy
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import Module
# from torch.nn.modules import MultiheadAttention
from torch.nn.modules import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules import Dropout
from torch.nn.modules import Linear
from torch.nn.modules import LayerNorm
from torch.nn.modules import Conv2d

from cdistnet.model.stage.multiheadAttention import MultiheadAttention

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class MDCDP(Module):
    r"""
     Multi-Domain CharacterDistance Perception
    """

    def __init__(self, decoder_layer, num_layers):
        super(MDCDP, self).__init__()

        d_model = 512
        self.num_layers = num_layers

        # step 1 SAE:
        self.layers_pos = _get_clones(decoder_layer, num_layers)

        # step 2 CBI:
        self.layers2 = _get_clones(decoder_layer, num_layers)
        self.layers3 = _get_clones(decoder_layer, num_layers)

        # step 3 :DSF
        self.dynamic_shared_fusion = DSF(d_model,2)

    def forward(self, sem, vis, pos, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        # print("start!!:pos{},\n sem{}\n vis{} \n".format(pos.shape,sem.shape,vis.shape))
        for i in range(self.num_layers):
            # step 1 : SAE
            # pos
            pos = self.layers_pos[i](pos, pos, pos,
                                    memory_mask=tgt_mask,
                                    memory_key_padding_mask=tgt_key_padding_mask)
            # print("pos:{}".format(pos.shape))


            #----------step 2 -----------: CBI
            # CBI-V : pos_vis
            pos_vis = self.layers2[i](pos, vis, vis,
                                    memory_mask=memory_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)
            # print("pos_vis:{}".format(pos_vis.shape))

            # CBI-S : pos_sem
            pos_sem = self.layers3[i](pos, sem, sem,
                                      memory_mask=tgt_mask,
                                      memory_key_padding_mask=tgt_key_padding_mask)
            # print("pos_sem:{}".format(pos_sem.shape))

            # ----------step 3 -----------: DSF
            pos = self.dynamic_shared_fusion(pos_vis, pos_sem)

        output = pos
        return output

class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.pos_embeding = nn.Parameter(torch.zeros(48,1, 512))
        self.pos_encoding = PositionalEncoding(dropout=0.0, dim=512)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None,pos_test=False):
        r"""Pass the input through the endocder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
            output: math(S,N,E)
        """
        # pos message for encoder
        # pos = src.new_zeros(*src.shape)
        # pos = self.pos_encoding(pos)
        if pos_test == True:
            pos = self.pos_embeding
        # print("src:{}".format(src.shape))
        # print("pos:{}".format(pos.shape))
            output = src + pos
        else:
            output = src
        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)
        if self.norm:
            output = self.norm(output)
        if src_key_padding_mask is not None:
            # only show no mask seq value
            output = output.permute(1, 0, 2) * torch.unsqueeze(~src_key_padding_mask, dim=-1).to(torch.float)
            output = output.permute(1, 0, 2)
            return output
        return output


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, attention_dropout_rate=0.0, residual_dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=attention_dropout_rate)
        # # # Implementation of Feedforward model
        # self.linear1 = Linear(d_model, dim_feedforward)
        # self.dropout = Dropout(attention_dropout_rate)
        # self.linear2 = Linear(dim_feedforward, d_model)

        self.conv1 = Conv2d(in_channels=d_model, out_channels=dim_feedforward, kernel_size=(1, 1))
        self.conv2 = Conv2d(in_channels=dim_feedforward, out_channels=d_model, kernel_size=(1, 1))
        # torch.nn.init.xavier_uniform_(self.conv1.weight.data)
        # torch.nn.init.xavier_uniform_(self.conv2.weight.data)
        # if self.conv1.bias is not None:
        #     if self.conv1.bias is not None:
        #         self.conv1.bias.data.zero_()
        # if self.conv2.bias is not None:
        #     if self.conv2.bias is not None:
        #         self.conv2.bias.data.zero_()

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(residual_dropout_rate)
        self.dropout2 = Dropout(residual_dropout_rate)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the endocder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # default
        # src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))

        src = src.permute(1, 2, 0)
        src = torch.unsqueeze(src, 2)
        src2 = self.conv2(F.relu(self.conv1(src)))
        src2 = torch.squeeze(src2, 2)
        src2 = src2.permute(2, 0, 1)
        src = torch.squeeze(src, 2)
        src = src.permute(2, 0, 1)

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class DSF(nn.Module):
    def __init__(self, d_model,fusion_num):
        super(DSF, self).__init__()
        self.w_att = nn.Linear(fusion_num * d_model, d_model)

    def forward(self, l_feature, v_feature):
        """
        Args:
            l_feature: (N, T, E) where T is length, N is batch size and d is dim of model
            v_feature: (N, T, E) shape the same as l_feature
            l_lengths: (N,)
            v_lengths: (N,)
        """
        f = torch.cat((l_feature, v_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * v_feature + (1 - f_att) * l_feature

        return output

class CommonAttentionLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, attention_dropout_rate=0.0, residual_dropout_rate=0.1):
        super(CommonAttentionLayer, self).__init__()

        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=attention_dropout_rate)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(attention_dropout_rate)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout2 = Dropout(residual_dropout_rate)
        self.dropout3 = Dropout(residual_dropout_rate)

    def forward(self, query, key, value, memory_mask=None,
                memory_key_padding_mask=None):
        """Pass the inputs (and mask) through the decoder layer.
        """

        out = self.multihead_attn(query, key, value, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        out = query + self.dropout2(out)
        out = self.norm2(out)

        out2 = self.linear2(self.dropout(F.relu(self.linear1(out))))
        out = out + self.dropout3(out2)
        out = self.norm3(out)
        return out

class CommonDecoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, attention_dropout_rate=0.0, residual_dropout_rate=0.1):
        super(CommonDecoderLayer, self).__init__()

        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=attention_dropout_rate)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(attention_dropout_rate)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.conv1 = Conv2d(in_channels=d_model, out_channels=dim_feedforward, kernel_size=(1, 1))
        self.conv2 = Conv2d(in_channels=dim_feedforward, out_channels=d_model, kernel_size=(1, 1))

        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout2 = Dropout(residual_dropout_rate)
        self.dropout3 = Dropout(residual_dropout_rate)

    def forward(self, query, key, value, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """Pass the inputs (and mask) through the decoder layer.
        """

        out = self.multihead_attn(query, key, value, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        out = query + self.dropout2(out)
        out = self.norm2(out)

        # default
        # tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = out.permute(1, 2, 0)
        tgt = torch.unsqueeze(tgt, 2)
        tgt2 = self.conv2(F.relu(self.conv1(tgt)))
        tgt2 = torch.squeeze(tgt2, 2)
        tgt2 = tgt2.permute(2, 0, 1)
        tgt = torch.squeeze(tgt, 2)
        tgt = tgt.permute(2, 0, 1)

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        # x(w,b,h*c)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PositionalEncoding_2d(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding_2d, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.avg_pool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(dim, dim)
        self.linear1.weight.data.fill_(1.)
        self.avg_pool_2 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear2 = nn.Linear(dim, dim)
        self.linear2.weight.data.fill_(1.)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        # x = x + self.pe[:x.size(0), :]
        w_pe = self.pe[:x.size(-1), :]
        w1 = self.linear1(self.avg_pool_1(x).squeeze()).unsqueeze(0)
        w_pe = w_pe * w1
        w_pe = w_pe.permute(1, 2, 0)
        w_pe = w_pe.unsqueeze(2)

        h_pe = self.pe[:x.size(-2), :]
        w2 = self.linear2(self.avg_pool_2(x).squeeze()).unsqueeze(0)
        h_pe = h_pe * w2
        h_pe = h_pe.permute(1, 2, 0)
        h_pe = h_pe.unsqueeze(3)

        x = x + w_pe + h_pe
        x = x.contiguous().view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)

        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, padding_idx, scale_embedding):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.embedding.weight.data.normal_(mean=0.0, std=d_model**-0.5)
        self.d_model = d_model
        self.scale_embedding = scale_embedding

    def forward(self, x):
        if self.scale_embedding:
            return self.embedding(x) * math.sqrt(self.d_model)
        return self.embedding(x)
