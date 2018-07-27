import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from ..attention import MultiHeadAttention
from ..utils import PositionWise


class Decoder(nn.Module):
    def __init__(self, embeddings, n_layers, n_heads, h_s, p_s, vocab_s, dropout=0.1):
        """
        :param embeddings: An instance of embeddings layer
        :param n_heads: Number of attention heads
        :param h_s: hidden size of input
        :param p_s: size of projected queries, keys and values
        :param vocab_s: vocab size
        :param dropout: drop prob
        """
        super(Decoder, self).__init__()

        self.embeddings = embeddings
        self.vocab_s = vocab_s
        self.layers = nn.ModuleList([
            DecoderLayer(n_heads, h_s, p_s, dropout)
            for _ in range(n_layers)
        ])

        self.out_to_vocab = nn.Sequential(
            nn.Linear(h_s, 1000),
            nn.SELU(),
            nn.Linear(1000, vocab_s)
        )

    def forward(self, input):
        """
        :param input: An long tensor with shape of [b_s, input_len]
        :return: An float tensor with shape of [b_s, input_len, h_s]
        """

        b_s, input_len = input.size()

        input, _ = self.embeddings(input)
        mask = self.autoregressive_mask(b_s, input_len, input.device)

        out = input
        for layer in self.layers:
            out = layer(out, mask)

        return self.out_to_vocab(out)

    def generate(self, seed, device):

        # 1 is go token idx
        seed_len = len(seed)
        input_idx = t.tensor([seed], dtype=t.long, device=device)

        input, _ = self.embeddings(input_idx)

        history = []
        for i, layer in enumerate(self.layers):
            history += [input]
            input = layer(input, mask=self.autoregressive_mask(1, len(seed), device))

        out = F.softmax(1.5 * self.out_to_vocab(input[:, -1]).squeeze(), dim=-1)

        result = []

        for step in range(self.embeddings.max_len - 2):

            out = out.cpu().numpy()
            input_idx = int(np.random.choice(self.vocab_s, 1, p=out)[0])
            if input_idx == 2:
                break

            result += [input_idx]

            input_idx = t.tensor([[input_idx]], dtype=t.long, device=device)
            pos = t.tensor([[step + 1 + seed_len]], dtype=t.long, device=device)
            input, _ = self.embeddings(input_idx, pos)

            for i, layer in enumerate(self.layers):
                history[i] = t.cat([history[i], input], 1)
                input = layer.inference_mode(input, history[i])

            out = F.softmax(1.5 * self.out_to_vocab(input).squeeze(), dim=-1)

        return result

    @staticmethod
    def autoregressive_mask(batch_size, length, device):
        mask = t.ones(length, length, dtype=t.uint8, device=device).tril_(-1)
        return mask.transpose(0, 1).repeat(batch_size, 1).view(batch_size, length, length)


class DecoderLayer(nn.Module):
    def __init__(self, n_heads, h_s, p_s, dropout=0.1):
        """
        :param n_heads: Number of attention heads
        :param h_s: hidden size of input
        :param p_s: size of projected queries and keys
        :param dropout: drop prob
        """
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(n_heads, h_s, p_s, dropout)
        self.position_wise = PositionWise(h_s, h_s * 4, dropout)

    def forward(self, input, mask):
        """
        :param input: An float tensor with shape of [b_s, input_len, h_s]
        :param mask: An byte tensor with shape of [b_s, input_len, input_len]
        :return: An float tensor with shape of [b_s, input_len, h_s]
        """

        result = self.attention(q=input, k=input, v=input, mask=mask)
        return self.position_wise(result)

    def inference_mode(self, q, input):
        """
        :param q: An float tensor with shape of [b_s, 1, h_s]
        :param input: An float tensor with shape of [b_s, input_len, h_s]
        :return: An float tensor with shape of [b_s, 1, h_s]
        """

        '''
        I believe that there is the way this module could be written in more efficient and beautiful way,
        in order to prevent having two different functions 
        for training and inference modes with quite similar semantics,
        but I'm not sure if I want to find it.
        '''

        result = self.attention(q=q, k=input, v=input)
        return self.position_wise(result)
