import torch as t
import torch.nn as nn

from nn.transformer import Decoder
from nn.utils import PosEmbedding


class Model(nn.Module):
    def __init__(self, n_l=6, n_h=8, d=0.1, **kwargs):
        """
        :param n_l: Number of attention layers
        :param n_h: Number of heads per attention layer
        :param d: dropout parameter
        """
        super(Model, self).__init__()

        self.embed = PosEmbedding(**kwargs)

        self.h_s = self.embed.h_s
        self.v_s = self.embed.v_s

        self.decoder = Decoder(self.embed, n_l, n_h, self.h_s, int(2 * self.h_s / n_h), self.v_s, d)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input, target):

        prediction = self.decoder(input)

        prediction = prediction.view(-1, self.v_s)
        target = target.view(-1)

        return self.criterion(prediction, target)

    def generate(self, seed, device):

        return self.decoder.generate(seed, device)

    def learnable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p
