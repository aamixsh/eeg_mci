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

from models import CNN, MLP
from data import EEGSDataset
from utils import *

parser = argparse.ArgumentParser(description='Run EEG Spectral experiments')
parser.add_argument('--num_channels', type=int, default=19)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--remove_channels', type=bool, default=True)
parser.add_argument('--rm_ch', type=str, default='14_15')
parser.add_argument('--meta_dir', type=str,default='./meta_data')
parser.add_argument('--results_dir', type=str,default='./results_split')
parser.add_argument('--weights_dir', type=str,default='./weights_split')
parser.add_argument('--data_path', type=str,default='./data/kmci_kctrl_kdem_psd')
parser.add_argument('--car', type=bool, default=True)
parser.add_argument('--sf', type=float, default=200.)
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
parser.add_argument('--model_type', type=str, default='mlp')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--model_conf_path', type=str, default='./grid_confs/conf')
parser.add_argument('--split', type=float, default=0.75)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--study_run', type=int, default=1)
parser.add_argument('--set_type', type=str, default='test')


args = parser.parse_args()

print('Args:')
for k, v in sorted(vars(args).items()):
  print("\t{}: {}".format(k, v))

# Set CUDA
if args.cuda:
    torch.cuda.set_device(args.gpu)

args.car_str = '_orig'
if args.car:
    args.car_str = '_car'

if args.study.startswith('k'):
    args.car_str = ''
    
# Get channels
rm_ch = set([])
args.rm_ch_str = ''
if args.remove_channels:
    rm_ch = set([int(i) for i in args.rm_ch.split('_')])
    args.rm_ch_str = args.rm_ch
    
channels = np.array([i for i in range(args.num_channels) if i not in rm_ch])

# Load data.
if args.rm_ch_str != '':
    data_path = args.data_path + '_' + args.rm_ch_str +'_notevt_' + str(args.contig_len) + args.car_str + '.pkl'
else:
    data_path = args.data_path +'_notevt_' + str(args.contig_len) + args.car_str + '.pkl'
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

conf_path = args.model_conf_path + '_' + args.model_type + '_' + str(len(channels)) + '_best.txt'

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
train_pats, test_pats = get_train_test_pats_from_run(f'{args.results_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}{args.car_str}.txt', args.study_run)
pats = {'train': train_pats, 'test': test_pats}

train_locs, test_locs = get_train_test_locs_from_pats(args, data, locs, pats)

interesting_pats = pickle.load(open(f'{args.set_type}_{args.study}_{args.treat}_interesting_patients_new_{args.study_run}.pkl', 'rb'))

collect_locs = []
for pat in interesting_pats:
    if args.set_type == 'test':
        collect_locs.extend([loc for loc in test_locs if str(loc[0]) + str(loc[1]) == pat])
    else:
        collect_locs.extend([loc for loc in train_locs if str(loc[0]) + str(loc[1]) == pat])

interesting_pats = []
int_inds = []
count = 0
for loc in collect_locs:
    if str(loc[0]) + str(loc[1]) not in interesting_pats:
        int_inds.append(count)
        print (count, str(loc[0]) + str(loc[1]))
        interesting_pats.append(str(loc[0]) + str(loc[1]))
        count = 0
    count += 1

print (count)
int_inds.append(count)

pickle.dump(collect_locs, open(f'{args.set_type}_{args.study}_{args.treat}_interesting_locs_new_{args.study_run}.pkl', 'wb'))

args.num_psd = data[collect_locs[0][0]][collect_locs[0][1]][collect_locs[0][2]].shape[1]

# For normalization
n = get_norms(args, data, channels, typs)

train_dataset = EEGSDataset(data_dict=data, locs=train_locs, study=args.study, treat=args.treat, norms=n)
dataset = EEGSDataset(data_dict=data, locs=collect_locs, study=args.study, treat=args.treat, norms=n)

# Get Dataloaders
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)


# Get Model
if args.model_type == 'mlp':
    model = MLP(num_channels=len(channels), num_psd=args.num_psd, out_dim=args.num_classes, conf=conf[1])
else:
    model = CNN(num_channels=len(channels), contig_len=args.contig_len, out_dim=args.num_classes, conf=conf[1])

if args.cuda:
    model = model.cuda()
print (model)

model.load_state_dict(torch.load(f'{args.weights_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{args.car_str}_{args.study_run}.pth'))

if args.cuda:
    model.cuda()
model.eval()


inputs = None
outputs = None

for idx, (X, y) in tqdm(enumerate(dataloader)):
    y_hat = model(X).softmax(dim=1)
    if inputs is not None:
        inputs = torch.cat((inputs, X.detach().cpu()))
        outputs = torch.cat((outputs, y_hat.detach().cpu()))
    else:
        inputs = X.detach().cpu()
        outputs = y_hat.detach().cpu()

print (inputs.shape)
print (outputs.shape)

pickle.dump(inputs, open(f'{args.set_type}_{args.study}_{args.treat}_interesting_inputs_new_{args.study_run}.pkl', 'wb'))
pickle.dump(outputs, open(f'{args.set_type}_{args.study}_{args.treat}_interesting_outputs_new_{args.study_run}.pkl', 'wb'))


train_inds = []
train_patss = []
count = 0
switch_ind = 0
for loc in train_locs:
    if str(loc[0]) + str(loc[1]) not in train_patss:
        train_inds.append(count)
        print (count, str(loc[0]) + str(loc[1]))
        train_patss.append(str(loc[0]) + str(loc[1]))
        count = 0
        if not switch_ind and loc[0] != args.study:
            switch_ind = len(train_inds)
    count += 1

train_inds.append(count)

int_inds = np.cumsum(int_inds)
train_inds = np.cumsum(train_inds)

print (int_inds)
print (train_inds)

print (switch_ind)
print (len(train_inds))

int_inputs = inputs.reshape(len(inputs), -1)
train_inputs = None
for idx, (X, y) in tqdm(enumerate(train_dataloader)):
    if train_inputs is not None:
        train_inputs = torch.cat((train_inputs, X.detach().cpu()))
    else:
        train_inputs = X.detach().cpu()

train_inputs = train_inputs.reshape(len(train_inputs), -1)
# int_inputs_0 = list(range(int_inds[2]))
# int_inputs_1 = list(range(int_inds[7], int_inds[9]))

# int_inputs_0 = int_inputs[int_inputs_0]
# int_inputs_1 = int_inputs[int_inputs_1]

print (int_inputs.shape, train_inputs.shape)

dists = torch.cdist(int_inputs, train_inputs)

print (dists.shape)

closest_train_inds = []

# iter_inds = [0, 1, 5, 6]
iter_inds = [0, 1, 2, 3]

for i in iter_inds:
    int_range = list(range(int_inds[i], int_inds[i + 1]))
    min_distance = np.inf
    for j in range(switch_ind, len(train_inds) - 1):
        train_range = list(range(train_inds[j], train_inds[j + 1]))
        dist_pat = dists[int_range, :][:, train_range].mean()
        print (f'{dist_pat.item():.2f}', end=' ')
        if dist_pat.item() < min_distance:
            min_distance = dist_pat.item()
            min_train_ind = j
    closest_train_inds.append(min_train_ind)
    print (min_distance, min_train_ind)

print ('0', closest_train_inds)

train_int_inputs = []
train_int_locs = []
train_lens = []
for i in range(len(closest_train_inds)):
    train_int_inputs.append(train_inputs[train_inds[closest_train_inds[i]]:train_inds[closest_train_inds[i] + 1], :])
    train_int_locs.extend(train_locs[train_inds[closest_train_inds[i]]:train_inds[closest_train_inds[i] + 1]])
    train_lens.append(train_inds[closest_train_inds[i] + 1] - train_inds[closest_train_inds[i]])

closest_train_inds = []
# iter_inds = [7, 8, 12, 13]
iter_inds = [6, 7, 8, 9]

for i in iter_inds:
    int_range = list(range(int_inds[i], int_inds[i + 1]))
    min_distance = np.inf
    for j in range(switch_ind - 1):
        train_range = list(range(train_inds[j], train_inds[j + 1]))
        dist_pat = dists[int_range, :][:, train_range].mean()
        print (f'{dist_pat.item():.2f}', end=' ')
        if dist_pat.item() < min_distance:
            min_distance = dist_pat.item()
            min_train_ind = j
    closest_train_inds.append(min_train_ind)
    print (min_distance, min_train_ind)

print ('1', closest_train_inds)

for i in range(len(closest_train_inds)):
    train_int_inputs.append(train_inputs[train_inds[closest_train_inds[i]]:train_inds[closest_train_inds[i] + 1], :])
    train_int_locs.extend(train_locs[train_inds[closest_train_inds[i]]:train_inds[closest_train_inds[i] + 1]])
    train_lens.append(train_inds[closest_train_inds[i] + 1] - train_inds[closest_train_inds[i]])


print (train_lens)

print (len(train_int_inputs))
print (len(train_int_locs))
print (train_int_inputs[0].shape)

train_int_inputs = torch.cat(train_int_inputs, dim=0)
print (train_int_inputs.shape)

pickle.dump(train_int_inputs, open(f'{args.set_type}_{args.study}_{args.treat}_closest_train_inputs_new_{args.study_run}.pkl', 'wb'))
pickle.dump(train_int_locs, open(f'{args.set_type}_{args.study}_{args.treat}_closest_train_locs_new_{args.study_run}.pkl', 'wb'))
pickle.dump(train_lens, open(f'{args.set_type}_{args.study}_{args.treat}_closest_train_lens_new_{args.study_run}.pkl', 'wb'))
