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

from models import CNN
from data import EEGDataset
from utils import *

parser = argparse.ArgumentParser(description='Run EEG Raw experiments')
parser.add_argument('--num_channels', type=int, default=19)     
parser.add_argument('--num_classes', type=int, default=2)
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
parser.add_argument('--num_epochs', type=int, default=10)
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
    torch.cuda.set_device(1)
    
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


# Get clean locations
locs = get_clean_locs(args, all_data, typs)

print (len(locs))

# Get data infdo
info = get_info(locs, args)

# Print class wise patient info.
print (info)

pats = {args.study: set(), args.treat: set()}
for typ in info:
    for pat, _ in info[typ]:
        pats[typ].add(pat)

for typ in info:
    total = 0
    for i, pat in enumerate(info[typ]):
        total += pat[1]
    print (typ, total / i, i)

print (pats)

data = all_data

# Get clean train_test locations
train_pats, test_pats = get_train_test_pats_from_run(f'{args.results_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}.txt', args.study_run)
pats = {'train': train_pats, 'test': test_pats}

train_locs, test_locs = get_train_test_locs_from_pats(args, data, locs, pats)

args.num_psd = data[train_locs[0][0]][train_locs[0][1]][train_locs[0][2]].shape[1]

# For normalization
n = get_norms(args, data, channels, typs)

train_dataset = EEGDataset(data_dict=data, locs=train_locs, study=args.study, treat=args.treat, norms=n)
test_dataset = EEGDataset(data_dict=data, locs=test_locs, study=args.study, treat=args.treat, norms=n)

# Get Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Get Model
if args.model_type == 'mlp':
    model = MLP(num_channels=len(channels), num_psd=args.num_psd, out_dim=args.num_classes, conf=conf[1])
else:
    model = CNN(num_channels=len(channels), contig_len=args.contig_len, out_dim=args.num_classes, conf=conf[1])

model.load_state_dict(torch.load(f'{args.weights_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{args.study_run}.pth'))

if args.cuda:
    model.cuda()
model.eval()

embeddings = None
for idx, (X, y) in tqdm(enumerate(test_dataloader)):
    encoding = model.conv_layers(X).view(X.shape[0], -1)
    if embeddings is not None:
        embeddings = torch.cat((embeddings, encoding.detach().cpu()))
    else:
        embeddings = encoding.detach().cpu()

print (embeddings.shape)

pickle.dump(embeddings, open(f'data/{args.study}_{args.treat}_{args.study_run}_test_embeddings.pkl', 'wb'))

embeddings = None
for idx, (X, y) in tqdm(enumerate(train_dataloader)):
    encoding = model.conv_layers(X).view(X.shape[0], -1)
    if embeddings is not None:
        embeddings = torch.cat((embeddings, encoding.detach().cpu()))
    else:
        embeddings = encoding.detach().cpu()

print (embeddings.shape)

pickle.dump(embeddings, open(f'data/{args.study}_{args.treat}_{args.study_run}_train_embeddings.pkl', 'wb'))