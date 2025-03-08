import argparse

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datasets
    parser.add_argument('--datadir', type=str, help='data directory')
    
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # Optimization options
    parser.add_argument('--epoch', default=20, type=int, metavar='N',
                        help='number of epochs to run')
    parser.add_argument('--train-batch', default=100, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--lr', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--optim', default='adam', type=str,
                        help='function to approximate')
    parser.add_argument('--lr_strategy', default='coslr', type=str, help='lr strategy: coslr constant')

    # parser.add_argument('--model', default='model', type=str, help='model')
    parser.add_argument('--input_size',  default=100,type=int, help='node size')
    parser.add_argument('--hidden_size',  default=100,type=int, help='node size')
    parser.add_argument('--output_size', default=100, type=int, help='node size')
    parser.add_argument('--layers',  default=4,type=int, help='layers')
    
    parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    # Checkpoints
    parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start_epoch', default=0, type=int)
    # Miscs
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    parser.add_argument('--model', default='weight', type=str)
    
    return parser