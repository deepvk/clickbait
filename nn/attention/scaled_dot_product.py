from math import sqrt

import torch as t
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, s):
        """
        :param s: int number that is necessary for estimation scaling factor – size of q and k
        :param dropout: drop prob
        """
        super(ScaledDotProductAttention, self).__init__()

        self.scaling = 1 / (sqrt(s))

    def forward(self, q, k, v, mask=None):
        """
        :param q: An float tensor with shape of [batch_size, query_len, s]
        :param k: An float tensor with shape of [batch_size, seq_len, s]
        :param v: An float tensor with shape of [batch_size, seq_len, value_s]
        :param mask: An byte tensor with shape of [batch_size, query_len, seq_len]
        :return: An float tensor with shape of [batch_size, query_len, value_s]
                     and attention map with shape of [batch_size, query_len, seq_len]
        """

        batch_size, query_len, _ = q.size()

        attention = t.bmm(q, k.transpose(1, 2)) * self.scaling

        '''
        In order to prevent contribution of padding symbols in attention lockup, 
        it is necessary to use attention mask
        '''
        if mask is not None:
            attention.data.masked_fill_(mask.data, -float('inf'))

        attention = F.softmax(attention, dim=2)

        return t.bmm(attention, v), attention
