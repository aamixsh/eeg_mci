import numpy as np
import os
import argparse
import pickle

from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn, optim, autograd
from torch.utils.data import DataLoader

from torcheval.metrics import MulticlassAccuracy, MulticlassAUPRC, MulticlassAUROC, MulticlassConfusionMatrix, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from torchmetrics.functional.classification import multiclass_calibration_error as mce

from models import CNN, RealNVP
from data import EEGDataset
from utils import *

parser = argparse.ArgumentParser(description='Run EEG Raw experiments')
parser.add_argument('--num_channels', type=int, default=19)     
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--remove_channels', type=bool, default=True)
parser.add_argument('--rm_ch', type=str, default='14_15')
parser.add_argument('--meta_dir', type=str,default='./meta_data')
parser.add_argument('--results_dir', type=str,default='./results_split')
parser.add_argument('--weights_dir', type=str,default='./weights_split')
parser.add_argument('--data_path', type=str,default='./data/kmci_kctrl_kdem')
parser.add_argument('--sf', type=float, default=250.)
parser.add_argument('--contig_len', type=int, default=200)
parser.add_argument('--art_thresh', type=int, default=0)
parser.add_argument('--lowcut', type=float, default=4.)
parser.add_argument('--highcut', type=float, default=40.)
parser.add_argument('--min_contigs', type=int, default=3)
parser.add_argument('--include_evt', type=bool, default=False)
parser.add_argument('--balanced', type=bool, default=False)
parser.add_argument('--norm_type', type=str, default='meanstd')
parser.add_argument('--study', type=str, default='kmci')
parser.add_argument('--treat', type=str, default='kctrl')
parser.add_argument('--model_type', type=str, default='cnn')
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--model_conf_path', type=str, default='./grid_confs/conf')
parser.add_argument('--split', type=float, default=0.75)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--study_run', type=int, default=12)


args = parser.parse_args()

print('Args:')
for k, v in sorted(vars(args).items()):
  print("\t{}: {}".format(k, v))

# Set CUDA
if args.cuda:
    torch.cuda.set_device(args.gpu)
    
# Get channels
rm_ch = set([])
args.rm_ch_str = ''
if args.remove_channels:
    rm_ch = set([int(i) for i in args.rm_ch.split('_')])
    args.rm_ch_str = args.rm_ch
    
channels = np.array([i for i in range(args.num_channels) if i not in rm_ch])

# Load data.
if args.rm_ch_str != '':
    data_path = args.data_path + '_' + args.rm_ch_str +'_notevt_' + str(args.contig_len) + '.pkl'
else:
    data_path = args.data_path +'_notevt_' + str(args.contig_len) + '.pkl'
all_data = pickle.load(open(data_path, 'rb'))

# create dirs
if not os.path.exists(args.meta_dir):
    os.makedirs(args.meta_dir)
if not os.path.exists(args.results_dir + '_' + args.rm_ch_str):
    os.makedirs(args.results_dir + '_' + args.rm_ch_str)
if not os.path.exists(args.weights_dir + '_' + args.rm_ch_str):
    os.makedirs(args.weights_dir + '_' + args.rm_ch_str)

    
# Define class types
typs = {args.study: 0, args.treat: 1}

conf_path = args.model_conf_path + '_' + args.model_type + '_' + str(args.contig_len) + '_' + str(len(channels)) + '_best.txt'

conf = read_grid_confs(conf_path)[0]

print ('Channels:', channels)

