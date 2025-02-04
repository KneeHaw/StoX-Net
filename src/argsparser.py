import argparse


def get_parser():

    parser = argparse.ArgumentParser(description='Hyperparameters for StoX-Net Training & Inference')

    ##################################################################################################################
    ## Regular Hyperparameters
    ##################################################################################################################
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    ##################################################################################################################
    ## Device Hyperparameters
    ##################################################################################################################
    # Convolution params
    parser.add_argument('--conv-time-steps', default=4, type=int, metavar='N',
                        help='Maximum time steps for each MTJ (default: 1)')
    parser.add_argument('--conv-num-ab', default=4, type=int, metavar='N',
                        help='Denotes maximum number of activation bits (default: 1)')
    parser.add_argument('--conv-num-wb', default=4, type=int, metavar='N',
                        help='Number of weight bits (default: 1)')
    parser.add_argument('--conv-ab-sw', default=4, type=int, metavar='N',
                        help='Number of activation bits per stream (default: 1)')
    parser.add_argument('--conv-wb-sw', default=4, type=int, metavar='N',
                        help='Number of weight bits per slices (default: 1)')

    # Linear params
    parser.add_argument('--linear-time-steps', default=4, type=int, metavar='N',
                        help='Maximum time steps for each MTJ (default: 1)')
    parser.add_argument('--linear-num-ab', default=4, type=int, metavar='N',
                        help='Denotes maximum number of activation bits (default: 1)')
    parser.add_argument('--linear-num-wb', default=4, type=int, metavar='N',
                        help='Number of weight bits (default: 1)')
    parser.add_argument('--linear-ab-sw', default=4, type=int, metavar='N',
                        help='Number of activation bits per stream (default: 1)')
    parser.add_argument('--linear-wb-sw', default=4, type=int, metavar='N',
                        help='Number of weight bits per slices (default: 1)')

    # General crossbar params
    parser.add_argument('--subarray-size', default=128, type=int, metavar='N',
                        help='Maximum subarray size for partial sums')
    parser.add_argument('--sensitivity', default=4, type=int, metavar='N',
                        help='Sensitivity \'a\' seen in tanh(ax) for MTJ modelling')

    ##################################################################################################################
    ## Saving/Loading Data
    ##################################################################################################################
    parser.add_argument('--resume', default='./saved/models/May30_constrictor_train_at_64bs_4sens.th', type=str, metavar='PATH',
                        help='absolute path to desired checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', default=False,
                        type=bool, help='evaluate model on validation set using resumed model')
    parser.add_argument('--model-save-dir', dest='model_save_dir',
                        help='The directory used to save the trained models',
                        default='./saved/models/test.th', type=str)
    parser.add_argument('--logs-save-dir', dest='logs_save_dir',
                        help='The directory used to save the trained models',
                        default='./saved/logs/test.txt', type=str)
    parser.add_argument('--save-every', dest='save_every',
                        help='Saves checkpoints at every specified number of epochs',
                        type=int, default=10)

    ##################################################################################################################
    ## Model and Dataset Parameters
    ##################################################################################################################
    parser.add_argument('--model', dest='model', help='Choose a model to run the network on'
                                                      '{resnet20, toymodel}', default='resnet20', type=str)
    parser.add_argument('--dataset', dest='dataset', help='Choose a dataset to run the network on from'
                                                          '{MNIST, CIFAR10}', default='MNIST', type=str)
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--stox', dest='stox',
                        help='Set to true if you want to use stochastic network, else FP',
                        default=True, type=bool)
    parser.add_argument('--MTJops-Exit', dest='MTJops_Exit',
                        help='Set to true if you just want to see number of MTJ operations in an inference',
                        default=False, type=bool)

    ##################################################################################################################
    ## Other parameters
    ##################################################################################################################
    parser.add_argument('--print-batch-info', dest='print_batch_info',
                        help='Set to true if you want to see per batch accuracy',
                        default=False, type=bool)
    parser.add_argument('--skip-to-batch', default=0, type=int,
                        metavar='N', help='Skip to this batch of images in inference')
    parser.add_argument('--batch-log', dest='batch_log',
                        help='Set log file for printing per-batch results',
                        default='./saved/batch_log.txt', type=str)
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 20)')

    return parser
