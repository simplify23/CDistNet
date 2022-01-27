import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter



class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model
        num_heads: parallel attention layers, or heads

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn


        self.conv1 = torch.nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 1))
        self.conv2 = torch.nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim * 2, kernel_size=(1, 1))
        self.conv3 = torch.nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim * 3, kernel_size=(1, 1))
        # torch.nn.init.xavier_uniform_(self.conv1.weight.data)
        # torch.nn.init.xavier_uniform_(self.conv2.weight.data)
        # torch.nn.init.xavier_uniform_(self.conv3.weight.data)
        # if self.conv1.bias is not None:
        #     if self.conv1.bias is not None:
        #         self.conv1.bias.data.zero_()
        # if self.conv2.bias is not None:
        #     if self.conv2.bias is not None:
        #         self.conv2.bias.data.zero_()
        # if self.conv3.bias is not None:
        #     if self.conv3.bias is not None:
        #         self.conv3.bias.data.zero_()
        self._reset_parameters()



    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight[:self.embed_dim, :])
        xavier_uniform_(self.in_proj_weight[self.embed_dim:(self.embed_dim * 2), :])
        xavier_uniform_(self.in_proj_weight[(self.embed_dim * 2):, :])

        xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """
        Inputs of forward function
            query: [target length, batch size, embed dim]
            key: [sequence length, batch size, embed dim]
            value: [sequence length, batch size, embed dim]
            key_padding_mask: if True, mask padding based on batch size
            attn_mask : triu mask for [T,T] or [T,S]
            incremental_state: if provided, previous time steps are cashed
            need_weights: output attn_output_weights
            static_kv: key and value are static

        Outputs of forward function
            attn_output: [target length, batch size, embed dim]
            attn_output_weights: [batch size, target length, sequence length]
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self._in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self._in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self._in_proj_kv(key)
        else:
            q = self._in_proj_q(query)
            k = self._in_proj_k(key)
            v = self._in_proj_v(value)
        # q *= self.scaling
        q = q*self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # q([batch*head,s_len,head_dim])
        # k([batch*head,t_len,head_dim])
        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        # step: q*k^T [batch*head,t_len,src_len]
        # attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        # sqrt(q*k^T)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))/math.sqrt(self.head_dim)
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # step: mask_triu
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            # [1,t_len,s_len]
            attn_output_weights += attn_mask

        # step: key_padding
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            # key_padding[batch,1,1,s_len]
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(
            attn_output_weights.float(), dim=-1,
            dtype=torch.float32 if attn_output_weights.dtype == torch.float16 else attn_output_weights.dtype)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads
        else:
            attn_output_weights = None

        return attn_output, attn_output_weights

    def _in_proj_qkv(self, query):
        # return self._in_proj(query).chunk(3, dim=-1)
        query = query.permute(1, 2, 0)
        query = torch.unsqueeze(query, dim=2)
        res = self.conv3(query)
        res = torch.squeeze(res, dim=2)
        res = res.permute(2, 0, 1)
        return res.chunk(3, dim=-1)

    def _in_proj_kv(self, key):
        # return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)
        key = key.permute(1, 2, 0)
        key = torch.unsqueeze(key, dim=2)
        res = self.conv2(key)
        res = torch.squeeze(res, dim=2)
        res = res.permute(2, 0, 1)
        return res.chunk(2, dim=-1)

    def _in_proj_q(self, query):
        # return self._in_proj(query, end=self.embed_dim)
        query = query.permute(1, 2, 0)
        query = torch.unsqueeze(query, dim=2)
        res = self.conv1(query)
        res = torch.squeeze(res, dim=2)
        res = res.permute(2, 0, 1)
        return res

    def _in_proj_k(self, key):
        # return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        key = key.permute(1, 2, 0)
        key = torch.unsqueeze(key, dim=2)
        res = self.conv1(key)
        res = torch.squeeze(res, dim=2)
        res = res.permute(2, 0, 1)
        return res

    def _in_proj_v(self, value):
        # return self._in_proj(value, start=2 * self.embed_dim)
        value = value.permute(1, 2, 0)
        value = torch.unsqueeze(value, dim=2)
        res = self.conv1(value)
        res = torch.squeeze(res, dim=2)
        res = res.permute(2, 0, 1)
        return res

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        # if bias is not None:
        #     bias = bias[start:end]
        return F.linear(input, weight, bias)