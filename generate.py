import argparse

import torch as t

from data.dataloader import Dataloader
from model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='persic')
    parser.add_argument('--seed', type=str, default='', metavar='S')
    parser.add_argument('--num-layers', type=int, default=8, metavar='NL',
                        help='num layers in decoder (default: 8)')
    parser.add_argument('--num-heads', type=int, default=14, metavar='NH',
                        help='num heads in each decoder layer (default: 14)')
    parser.add_argument('--state-dict', type=str, default='', metavar='SD',
                        help='Path to saved state dict')
    args = parser.parse_args()

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    loader = Dataloader('./data/')

    model = Model(args.num_layers,
                  args.num_heads,
                  0.,
                  max_len=loader.max_len,
                  embeddings_path='./data/embeddings.npy')
    model.load_state_dict(t.load(args.state_dict))
    model.to(device)

    model.eval()
    with t.no_grad():
        seed = [1] + loader.sp.EncodeAsIds(args.seed)

        generations = '\n'.join([loader.sp.DecodeIds(seed + model.generate(seed, device)) for i in range(100)])
        print(generations)
