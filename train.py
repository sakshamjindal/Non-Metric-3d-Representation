import argparse
import os

from core.trainer import run_training

parser = argparse.ArgumentParser(description='Relational 2d Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to datasets root directory')
parser.add_argument('-j', '--num-worker', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=5, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print iter frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument("--gpu", type=int, nargs='+', default=None, help='GPU id to use.')
parser.add_argument('--warmup-epoch', default=5, type=int,
                    help='number of warm-up epochs to only train with InfoNCE loss')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--exp-dir', default='experiment_pcl', type=str,
                    help='experiment directory to store tb logs and checkpoints')
parser.add_argument('--num-cluster', default='50,100,200', type=str, 
                    help='number of clusters (should be less than equal to number of samples)')
parser.add_argument('--temperature', default=0.2, type=float,help='softmax temperature')
parser.add_argument('--ret_freq', default=5, type=int,metavar='N', help='visualize epoch frequency (default: 5)')


if not os.path.exists('tb_logs'):
    os.makedirs('tb_logs')

args = parser.parse_args()

run_training(args)