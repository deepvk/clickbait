import numpy as np
import torch as t
import torch.nn as nn


class PosEmbedding(nn.Module):
    def __init__(self, **kwargs):
        super(PosEmbedding, self).__init__()

        self.max_len = kwargs['max_len']

        path = kwargs.get('embeddings_path', None)
        if path is not None:

            embeddings = np.load(path)
            self.v_s, self.h_s = embeddings.shape

            self.embeddings = nn.Embedding(self.v_s, self.h_s, padding_idx=0)
            self.embeddings.weight = nn.Parameter(t.from_numpy(embeddings).float(), requires_grad=False)

        else:
            self.v_s, self.h_s = kwargs['vocab_size'], kwargs['embedding_size']
            self.embeddings = nn.Embedding(self.v_s, self.h_s, padding_idx=0)

        self.positional_embeddings = nn.Embedding(self.max_len, self.h_s, padding_idx=0)
        self.position_encoding_init()

    def forward(self, input, positional=None):
        """
        :param input: An long tensor with shape of [b_s, s_l]
        :param positional: An long tensor with positional information with shape of [b_s, s_l]
        :return: An float tensor with shape of [b_s, s_l, embedding_s] and byte tensor mask with shape of [b_s, s_l]
        """

        b_s, s_l = input.size()

        if positional is None:
            '''
            It is necessary to provide positional information explicitly during autoregressive response generation
            '''
            positional = t.arange(1, s_l + 1, dtype=t.long, device=input.device).repeat(b_s).view(b_s, -1)

        mask = t.eq(input, 0)
        positional.masked_fill_(mask, 0)

        return self.embeddings(input) + self.positional_embeddings(positional), mask

    def position_encoding_init(self):
        encoding = np.array([
            [pos / np.power(10000, 2 * i / self.h_s) for i in range(self.h_s)]
            if pos != 0 else np.zeros(self.h_s) for pos in range(self.max_len)])

        encoding[1:, 0::2] = np.sin(encoding[1:, 0::2])
        encoding[1:, 1::2] = np.cos(encoding[1:, 1::2])
        encoding[0, :] = np.zeros(self.h_s)

        self.positional_embeddings.weight = nn.Parameter(t.tensor(encoding, dtype=t.float), requires_grad=False)
