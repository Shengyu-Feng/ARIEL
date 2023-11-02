import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset')
    parser.add_argument('--local', dest='local', action='store_const', 
            const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', 
            const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=False)

    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--eps', dest='eps', type=float, default=0.5,
            help='Adversarial coefficient.')
    parser.add_argument('--alpha', dest='alpha', type=int, default=200,
            help='Edge perturb ratio.')
    parser.add_argument('--beta', dest='beta', type=float, default=0.01,
        help='Feature perturb ratio.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
            help='')

    parser.add_argument('--aug', type=str, default='dnodes')
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()

