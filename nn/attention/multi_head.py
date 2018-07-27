import torch as t
import torch.nn as nn
from torch.nn.init import kaiming_normal_

from .scaled_dot_product import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, h_s, p_s, dropout=0.1):
        """
        :param n_heads: Number of attention heads
        :param h_s: hidden size of input
        :param p_s: size of projected queries, keys and values
        :param dropout: drop prob
        """
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.h_size = h_s

        self.q_proj = nn.Parameter(t.FloatTensor(n_heads, h_s, p_s))
        self.k_v_proj = nn.Parameter(t.FloatTensor(2 * n_heads, h_s, p_s))
        for param in [self.q_proj, self.k_v_proj]:
            kaiming_normal_(param.data)

        self.attention = ScaledDotProductAttention(p_s)

        self.out = nn.Linear(n_heads * p_s, h_s)
        self.layer_norm = nn.LayerNorm(h_s, eps=1e-12)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        :param q: An float tensor with shape of [batch_size, query_len, h_s]
        :param k: An float tensor with shape of [batch_size, seq_len, h_s]
        :param v: An float tensor with shape of [batch_size, seq_len, h_s]
        :param mask: An byte tensor with shape of [batch_size, query_len, seq_len]
        :return: An float tensor with shape of [batch_size, query_len, h_s]
        """

        batch_size = q.size(0)

        q_len = q.size(1)
        seq_len = k.size(1)

        residual = q

        q = self.repeat_n_heads(q)
        k = self.repeat_n_heads(k)
        v = self.repeat_n_heads(v)
        k_v = t.cat([k, v], 0)

        q = t.bmm(q, self.q_proj).view(-1, q_len, self.q_proj.size(2))
        k, v = t.split(t.bmm(k_v, self.k_v_proj), self.n_heads, 0)
        k = k.view(-1, seq_len, self.k_v_proj.size(2))
        v = k.view(-1, seq_len, self.k_v_proj.size(2))

        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)

        result, _ = self.attention(q, k, v, mask)
        result = t.split(result, batch_size, dim=0)
        result = t.cat(result, dim=-1)

        result = self.out(result)
        result = self.dropout(result)

        return self.layer_norm(result + residual)

    def repeat_n_heads(self, input):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, hidden_size]
        :return: An float tensor with shape of [n_heads, batch_size * seq_len, hidden_size]
        """
        return input.repeat(self.n_heads, 1, 1).view(self.n_heads, -1, self.h_size)
