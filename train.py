import argparse

import torch as t
from tensorboardX import SummaryWriter
from torch.optim import Adam as Optimizer

from data.dataloader import Dataloader
from model import Model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='clickbait')
    parser.add_argument('--num-iterations', type=int, default=70_000, metavar='NI',
                        help='num iterations (default: 70_000)')
    parser.add_argument('--batch-size', type=int, default=50, metavar='S')
    parser.add_argument('--num-threads', type=int, default=4, metavar='BS',
                        help='num threads (default: 4)')
    parser.add_argument('--num-layers', type=int, default=8, metavar='NL',
                        help='num layers in decoder (default: 8)')
    parser.add_argument('--num-heads', type=int, default=14, metavar='NH',
                        help='num heads in each decoder layer (default: 14)')
    parser.add_argument('--dropout', type=float, default=0.4, metavar='D',
                        help='dropout rate (default: 0.4)')
    parser.add_argument('--tensorboard', type=str, default='default_tb', metavar='TB',
                        help='Name for tensorboard model')
    args = parser.parse_args()

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    writer = SummaryWriter(args.tensorboard)

    t.set_num_threads(args.num_threads)
    loader = Dataloader('./data/')

    model = Model(args.num_layers,
                  args.num_heads,
                  args.dropout,
                  max_len=loader.max_len,
                  embeddings_path='./data/embeddings.npy')
    model.to(device)

    optimizer = Optimizer(model.learnable_parameters(), lr=0.0002, amsgrad=True)

    print('Model have initialized')

    for i in range(args.num_iterations):

        optimizer.zero_grad()

        model.train()

        input, target = loader.next_batch(args.batch_size, 'train', device)
        nll = model(input, target)
        nll.backward()

        optimizer.step()

        model.eval()

        if i % 100 == 0:
            input, target = loader.next_batch(args.batch_size, 'test', device)
            with t.no_grad():
                test_nll = model(input, target)

                writer.add_scalar('train loss', nll.cpu(), i)
                writer.add_scalar('test loss', test_nll.cpu(), i)
                print('i {}, train {} test {}'.format(i, nll.item(), test_nll.item()))
                print('_________')

        if i % 20 == 0:
            with t.no_grad():
                generation = model.generate([1], device)
                print(loader.sp.DecodeIds(generation))

        if (i + 1) % 10000 == 0:
            model = model.cpu()
            t.save(model.state_dict(), '{}_{}'.format(args.tensorboard, i))
            model.to(device)

    model.eval()
    with t.no_grad():
        generations = '\n'.join([loader.sp.DecodeIds(model.generate([1], device)) for i in range(5000)])
        with open('titles.txt', 'w') as f:
            f.write(generations)
