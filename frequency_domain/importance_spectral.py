import numpy as np
import pandas as pd
import os
import argparse
import pickle
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from tqdm import tqdm
from scipy import signal
from decimal import Decimal

import torch
from torch import nn, optim, autograd
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.nn import functional as F

from utils import *
from data import EEGDataset, EEGSDataset
from models import CNN, CNNS, CNNs, MLPS

from torcheval.metrics import MulticlassAccuracy, MulticlassAUPRC, MulticlassAUROC, MulticlassConfusionMatrix, MulticlassF1Score, MulticlassPrecision, MulticlassRecall


parser = argparse.ArgumentParser(description='Run EEG experiments')
parser.add_argument('--safe', type=bool, default=True)
parser.add_argument('--num_channels', type=int, default=19)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--remove_channels', type=str, default='14_15')
parser.add_argument('--meta_dir', type=str,default='meta_data_notevt')
parser.add_argument('--weights_dir', type=str,default='weights_spectral_notevt')
parser.add_argument('--results_dir', type=str,default='results_spectral_notevt')
parser.add_argument('--data_path', type=str,default='./mci_ctrl_ctrlb_psd_notevt.pkl')
parser.add_argument('--mask_path', type=str,default='./mask_notevt.npy')
parser.add_argument('--sf', type=float, default=250.)
parser.add_argument('--contig_len', type=int, default=250)
parser.add_argument('--art_thresh', type=int, default=0)
parser.add_argument('--lowcut', type=float, default=4.)
parser.add_argument('--highcut', type=float, default=40.)
parser.add_argument('--min_contigs', type=int, default=5)
parser.add_argument('--balanced', type=bool, default=False)
parser.add_argument('--start_evt', type=bool, default=True)
parser.add_argument('--evt_starts', type=str, default='1_2')
parser.add_argument('--filter_type', type=str, default='global')
parser.add_argument('--include_evt', type=bool, default=False)
parser.add_argument('--shuffle', type=str, default='patient')
parser.add_argument('--num_psd', type=int, default=26)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--study', type=str, default='mci')
parser.add_argument('--treat', type=str, default='ctrlb')
parser.add_argument('--num_epochs', type=int, default=75)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--kernel', type=int, default=8)
parser.add_argument('--depth', type=int, default=3)
parser.add_argument('--split', type=float, default=0.8)
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--selected_run', type=int, default=4)
parser.add_argument('--weight_runs', type=int, default=10)
parser.add_argument('--all', type=bool, default=True)

args = parser.parse_args()

print('Args:')
for k, v in sorted(vars(args).items()):
  print("\t{}: {}".format(k, v))

if args.cuda:
    torch.cuda.set_device(1)

# Load data.
data_filename = args.data_path.split('.')[-2]
# assert data_filename.endswith(args.remove_channels)

if not os.path.exists(args.meta_dir):
    os.makedirs(args.meta_dir)
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)
if not os.path.exists(args.weights_dir):
    os.makedirs(args.weights_dir)

all_data = pickle.load(open(args.data_path, 'rb'))

mask = None
if args.safe:
    mask = torch.tensor(np.load(args.mask_path)).type(torch.float32)

# Get channels
rm_ch = []
if args.remove_channels in args.data_path:
    rm_ch = set([int(i) for i in args.remove_channels.split('_')])
channels = np.array([i for i in range(args.num_channels) if i not in rm_ch])

print (channels)

in_dim = len(channels) * args.num_psd
if mask is not None:
    in_dim = np.count_nonzero(mask)
    mask = torch.reshape(mask, (len(channels), args.num_psd))

# EVT start vals
evts = [int(i) for i in args.evt_starts.split('_')]


print (args.study, args.shuffle, args.filter_type, args.start_evt, args.include_evt)

balanced = ''
if args.balanced:
    balanced = '_bal'

safe_str = ''
if args.safe:
    safe_str = '_True'

# Define class types
typs = {args.study: 0, args.treat: 1}

# Get clean locations
locs = get_clean_spectral_locs(args, all_data, typs, evts, channels)

print (len(locs))

# Get data info
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


# file = open(f'{args.results_dir}/{args.study}_{args.lr}_{args.num_epochs}_{args.start_evt}_{args.filter_type}_{args.shuffle}_{args.include_evt}{balanced}.txt', 'a')

for run in range(args.runs):
    for weight_run in range(args.weight_runs):

        data = all_data
        # Get clean train_test locations
        train_locs, test_locs = get_spectral_train_test_locs(args, data, locs, balanced)

        train_pats = {args.study: set(), args.treat: set()}
        test_pats = {args.study: set(), args.treat: set()}

        for loc in train_locs:
            train_pats[loc[0]].add(loc[1])
        for loc in test_locs:
            test_pats[loc[0]].add(loc[1])
        print (train_pats)
        print (test_pats)

        # For normalization
        channel_maxs = np.full((len(channels)), -np.inf)
        channel_mins = np.full((len(channels)), np.inf)
        channel_means = np.zeros(len(channels))
        channel_stds = np.zeros(len(channels))

        count = 0

        for typ in data:
            for pid in tqdm(range(len(data[typ]))):
                if not len(data[typ][pid]):
                    continue
                for i in range(len(channels)):
                    pmin = np.min(data[typ][pid][:, i, :])
                    pmax = np.max(data[typ][pid][:, i, :])
                    pstd = np.std(data[typ][pid][:, i, :])
                    pmean = np.mean(data[typ][pid][:, i, :])
                    if pmin < channel_mins[i]:
                        channel_mins[i] = pmin
                    if pmax > channel_maxs[i]:
                        channel_maxs[i] = pmax
                    channel_means[i] += pmean
                    channel_stds[i] += pstd
            count += len(data[typ])
        channel_means /= count
        channel_stds /= count

        print (channel_maxs)
        print (channel_mins)
        # print (channel_means)
        # print (channel_stds)
        norms = {'mean': torch.tensor(channel_means).type(torch.float32), 'std': torch.tensor(channel_stds).type(torch.float32)}
        # norms = {'mean': channel_means, 'std': channel_stds}


        train_batch_size = args.batch_size
        test_batch_size = 32
        n = None
        if args.normalize:
            n = norms

        train_dataset = EEGSDataset(data_dict=data, locs=train_locs, study=args.study, treat=args.treat, norms=n, evt=args.include_evt, cuda=False, mask=mask)
        test_dataset = EEGSDataset(data_dict=data, locs=test_locs, study=args.study, treat=args.treat, norms=n, evt=args.include_evt, cuda=False, mask=mask)

        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


        mlp = MLPS(num_channels=len(channels), num_psd=args.num_psd, out_dim=args.num_classes)

        mlp.load_state_dict(torch.load(f'{args.weights_dir}/{args.study}_{args.treat}_{args.remove_channels}_{args.start_evt}{safe_str}_{weight_run}_split.pth'))

        background = None

        for X, y in train_dataloader:
            # select backgroud for shap
            if background is not None and background.shape[0] < 1000:
                background = torch.cat((background, X))
            elif background is None:
                background = X
            else:
                break

        # DeepExplainer to explain predictions of the model
        explainer = shap.DeepExplainer(mlp, background)

#         print (background.mean(0).numpy().shape)
#         plt.imshow(background.mean(0).numpy())
#         plt.savefig('temp.png')
#         input()


        targets = None
        predictions = None

        imps = np.zeros((len(test_dataset), test_dataset[0][0].shape[0]))
        neg_imps = np.zeros((len(test_dataset), test_dataset[0][0].shape[0]))

        allt = ''
        if args.all:
            imps = np.zeros((len(test_dataset), test_dataset[0][0].shape[0], test_dataset[0][0].shape[1]))
            neg_imps = np.zeros((len(test_dataset), test_dataset[0][0].shape[0], test_dataset[0][0].shape[1]))
            allt = '_all'

#         file = open(f'{args.results_dir}/shap_imps{allt}_{args.study}_{args.treat}_{args.remove_channels}_{args.lr}_{args.num_epochs}_{args.start_evt}_{args.filter_type}_{args.shuffle}_{args.include_evt}{balanced}_{run}.txt', 'w')
#         filen = open(f'{args.results_dir}/shap_negimps{allt}_{args.study}_{args.treat}_{args.remove_channels}_{args.lr}_{args.num_epochs}_{args.start_evt}_{args.filter_type}_{args.shuffle}_{args.include_evt}{balanced}_{run}.txt', 'w')

        iter = 0
        for X, y in tqdm(test_dataloader):
            logits = mlp(X)
            if predictions is not None:
                predictions = torch.cat((predictions, logits.detach().cpu()))
            else:
                predictions = logits.detach().cpu()
            if targets is not None:
                targets = torch.cat((targets, y.cpu()))
            else:
                targets = y.cpu()

            # compute shap values
            shap_values = explainer.shap_values(X)
            for i in range(len(X)):
                if y[i]:
                    if args.all:
                        imps[iter * test_batch_size + i] = shap_values[1][i]
                        neg_imps[iter * test_batch_size + i] = shap_values[0][i]
                    else:
                        imps[iter * test_batch_size + i] = shap_values[1][i].sum(1)
                        neg_imps[iter * test_batch_size + i] = shap_values[0][i].sum(1)
                else:
                    if args.all:
                        imps[iter * test_batch_size + i] = shap_values[0][i]
                        neg_imps[iter * test_batch_size + i] = shap_values[1][i]
                    else:
                        imps[iter * test_batch_size + i] = shap_values[0][i].sum(1)
                        neg_imps[iter * test_batch_size + i] = shap_values[1][i].sum(1)

#                 for s in imps[iter * test_batch_size + i]:
#                     file.write(str(s)+',')
#                 file.write('\n')
#                 for s in neg_imps[iter * test_batch_size + i]:
#                     filen.write(str(s)+',')
#                 filen.write('\n')
            iter += 1

#         file.close()
#         filen.close()

        np.save(f'{args.results_dir}/shap_imps{allt}_{args.study}_{args.treat}_{args.remove_channels}_{args.start_evt}{safe_str}_{weight_run}', imps)
        np.save(f'{args.results_dir}/shaps_negimps{allt}_{args.study}_{args.treat}_{args.remove_channels}_{args.start_evt}{safe_str}_{weight_run}', neg_imps)


#         print (torch.count_nonzero(predictions.argmax(1) == targets) / len(predictions), len(predictions))


#         temp_dataset = EEGSDataset(data_dict=data, locs=test_locs, study=args.study, treat=args.treat, norms=n, evt=args.include_evt, cuda=False)
#         temp_loader = DataLoader(temp_dataset, batch_size=1, shuffle=False)

#         count_0 = 0
#         count_1 = 0
#         count_pred_0 = 0
#         count_pred_1 = 0
#         prev_pid = 0

#         accs = []

#         pat_results = {
#                     args.study: {},
#                     args.treat: {}
#                     }

#         acc = 0

#         for i, (X, y) in enumerate(temp_loader):
#             if test_locs[i][1] not in pat_results[test_locs[i][0]]:
#                 pat_results[test_locs[i][0]][test_locs[i][1]] = {'c': 0, 't': 0}
#             logits = mlp(X)
#             acc = mean_accuracy(logits, y)
#             pat_results[test_locs[i][0]][test_locs[i][1]]['t'] += 1
#             if acc == 1:
#                 pat_results[test_locs[i][0]][test_locs[i][1]]['c'] += 1

#         for typ in pat_results:
#             for pat in pat_results[typ]:
#                 pat_results[typ][pat]['acc'] = pat_results[typ][pat]['c'] / pat_results[typ][pat]['t']

#         print (pat_results)

#         total_acc = 0
#         total_count = 0
#         pat_predictions = []
#         pat_targets = []
#         for typ in pat_results:
#             typ_acc = 0
#             typ_count = 0
#             for pat in pat_results[typ]:
#                 true = typs[typ]
#                 other = 1 if true == 0 else 0
#                 pat_targets.append(true)
#                 pred = [0., 0.]
#                 if pat_results[typ][pat]['acc'] >= 0.5:
#                     typ_acc += 1
#                 pred[true] = pat_results[typ][pat]['acc']
#                 pred[other] = 1 - pat_results[typ][pat]['acc']
#                 pat_predictions.append(pred)
#                 typ_count += 1
#             print (typ, typ_acc, typ_count, typ_acc / typ_count)
#             total_acc += typ_acc
#             total_count += typ_count
#         print (total_acc, total_count, total_acc / total_count)

#         predictions = torch.tensor(pat_predictions)
#         targets = torch.tensor(pat_targets)

#         print (torch.count_nonzero(predictions.argmax(1) == targets) / len(predictions), len(predictions))
