import numpy as np
import sentencepiece as spm
import torch as t


class Dataloader:
    def __init__(self, data_path=''):
        """
        :param data_path: path to data
        """

        assert isinstance(data_path, str), \
            'Invalid data_path type. Required {}, but {} found'.format(str, type(data_path))

        self.targets = ['train', 'test']

        self.data = {
            target: np.load('{}{}.npy'.format(data_path, target))
            for target in self.targets
        }

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load('{}wut.model'.format(data_path))

        '''
        Actually, max length is lower than this value
        '''
        self.max_len = max([len(line) for tar in self.targets for line in self.data[tar]])

    def next_batch(self, batch_size, target, device):
        data = self.data[target]

        input = data[np.random.randint(len(data), size=batch_size)]

        input = [[1] + self.sp.EncodeAsIds(line) + [2] for line in input]

        target = self.padd_sequences([line[1:] for line in input])
        input = self.padd_sequences([line[:-1] for line in input])

        return tuple([t.tensor(i, dtype=t.long, device=device) for i in [input, target]])

    @staticmethod
    def padd_sequences(lines):
        lengths = [len(line) for line in lines]
        max_length = max(lengths)

        return np.array([line + [0] * (max_length - lengths[i]) for i, line in enumerate(lines)])
