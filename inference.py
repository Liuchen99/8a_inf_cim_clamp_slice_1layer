import argparse
import csv
import os
import random
import sys
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import trange
import torchvision.datasets as datasets
#import flops_benchmark
from clr import CyclicLR
from data import get_loaders
from logger import CsvLogger
#from model import MobileNet2
from resnet import resnet12_1w8a
from run import train, test, save_checkpoint, find_bounds_clr
import torchvision.transforms as transforms

from layers_1to2_newboard import layer1


import numpy as np
import pickle
from sklearn.metrics import confusion_matrix

torch.set_printoptions(profile="full")
torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(threshold=sys.maxsize, suppress=True)

parser = argparse.ArgumentParser(description='MobileNetv2 training with PyTorch')
parser.add_argument('--dataroot', default='/share/dataset/cifar/cifar10', metavar='/share/dataset/cifar/cifar10',
                    help='Path to ImageNet train and val folders, preprocessed as described in '
                         'https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset')
parser.add_argument('--gpus', default='0,1', help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='Number of data loading workers (default: 4)')
parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')

# Optimization options
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=4e-5, help='Weight decay (L2 penalty).')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma at scheduled epochs.')
parser.add_argument('--schedule', type=int, nargs='+', default=[200, 300],
                    help='Decrease learning rate at these epochs.')

# CLR
parser.add_argument('--clr', dest='clr', action='store_true', help='Use CLR')
parser.add_argument('--min-lr', type=float, default=1e-5, help='Minimal LR for CLR.')
parser.add_argument('--max-lr', type=float, default=1, help='Maximal LR for CLR.')
parser.add_argument('--epochs-per-step', type=int, default=20,
                    help='Number of epochs per step in CLR, recommended to be between 2 and 10.')
parser.add_argument('--mode', default='triangular2', help='CLR mode. One of {triangular, triangular2, exp_range}')
parser.add_argument('--find-clr', dest='find_clr', action='store_true',
                    help='Run search for optimal LR in range (min_lr, max_lr)')

# Checkpoints
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='Just evaluate model')
parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='Directory to store results')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='Number of batches between log messages')
parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: 1)')

# Architecture
parser.add_argument('--scaling', type=float, default=1, metavar='SC', help='Scaling of MobileNet (default x1).')
parser.add_argument('--input-size', type=int, default=224, metavar='I',
                    help='Input size of MobileNet, multiple of 32 (default 224).')


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict




def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpus:
        torch.cuda.manual_seed_all(args.seed)

    if args.gpus is not None:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        device = 'cuda:' + str(0)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True
    else:
        device = 'cpu'
    cudnn.benchmark = True
    print("---------",device)
    if args.type == 'float64':
        dtype = torch.float64
    elif args.type == 'float32':
        dtype = torch.float32
    elif args.type == 'float16':
        dtype = torch.float16
    else:
        raise ValueError('Wrong type!')  # TODO int8



    model = resnet12_1w8a()#(input_size=args.input_size, scale=args.scaling)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    print([l.nelement() for l in model.parameters()])
    # print(model)
    print('number of parameters: {}'.format(num_parameters))


    if args.gpus is not None:
        model = torch.nn.DataParallel(model)
    model.to(device=device, dtype=dtype)

    data = None
    print(device)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = 0 # checkpoint['epoch'] - 1
            best_test = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        elif os.path.isdir(args.resume):
            checkpoint_path = os.path.join(args.resume, 'checkpoint.pth.tar')
            csv_path = os.path.join(args.resume, 'results.csv')
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=device)
            args.start_epoch = 0 # checkpoint['epoch'] - 1
            best_test = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
            data = []
            with open(csv_path) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(row)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    y_out = np.zeros(50)
    y_label = np.zeros(10000)

    testXtr = unpickle("./test_batch")
    layer1_right = np.load('debug/result_of_50images_prototype.npy')

    for i in range(0, 50):
        img = np.reshape(testXtr['data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        # print(str(testXtr['labels'][i]) + '_' + str(i), np.max(img))

        # print((img/255).transpose([2, 0, 1]))
        # img_liu = (img/255).transpose([2, 0, 1])

        img_data = transforms.ToTensor()(img)

        channel_mean = [0.485, 0.456, 0.406]
        channel_std = [0.229, 0.224, 0.225]
        # for i in range(3):
        #     img_liu[i, :, :] = (img_liu[i, :, :] - channel_mean[i]) / channel_std[i]
        img_data = transforms.Normalize(channel_mean, channel_std)(img_data)

        img_data = torch.reshape(img_data, (1,3,32,32))
        model.eval()

        img_data = img_data.to(device=device, dtype=dtype)

        with torch.no_grad():

            # input_features = torch.from_numpy(np.load('layer1_result.npy')).view(1, 16, 32, 32)

            input_features = torch.from_numpy(layer1[i]).reshape([1, 16, 32, 32])

            input_features = input_features.type(torch.cuda.FloatTensor)
            output = model(input_features)
            print(output)
        y_label[i] = testXtr['labels'][i]
        y_out[i] = np.argmax(output.cpu().numpy(), axis = 1)
    print(y_out == layer1_right)
    print('chip', y_out)
    print('right:', layer1_right)
    # np.save('result_of_50images_prototype', y_out)


if __name__ == '__main__':
    main()



