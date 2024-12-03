import numpy as np
import os
import argparse
import pickle

from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import shap

import torch
from torch import nn, optim, autograd
from torch.utils.data import DataLoader

from torcheval.metrics import MulticlassAccuracy, MulticlassAUPRC, MulticlassAUROC, MulticlassConfusionMatrix, MulticlassF1Score, MulticlassPrecision, MulticlassRecall

from models import CNN, MLP
from data import EEGSDataset
from utils import *

parser = argparse.ArgumentParser(description='Run EEG Spectral cross-domain experiments')
parser.add_argument('--num_channels', type=int, default=19)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--remove_channels', type=bool, default=True)
parser.add_argument('--rm_ch', type=str, default='14_15')
parser.add_argument('--meta_dir', type=str,default='./meta_data_cross')
parser.add_argument('--results_dir', type=str,default='./results_split_cross')
parser.add_argument('--weights_dir', type=str,default='./weights_split_cross')
parser.add_argument('--data_path1', type=str,default='./data/kmci_kctrl_kdem_psd')
parser.add_argument('--data_path2', type=str,default='./data/wmci_wctrl_psd')
parser.add_argument('--car', type=bool, default=True)
parser.add_argument('--sf', type=float, default=200.)
parser.add_argument('--contig_len', type=int, default=200)
parser.add_argument('--art_thresh', type=int, default=0)
parser.add_argument('--lowcut', type=float, default=4.)
parser.add_argument('--highcut', type=float, default=40.)
parser.add_argument('--min_contigs', type=int, default=3)
parser.add_argument('--include_evt', type=bool, default=False)
parser.add_argument('--balanced', type=bool, default=True)
parser.add_argument('--norm_type', type=str, default='meanstd')
parser.add_argument('--study', type=str, default='kdem')
parser.add_argument('--treat', type=str, default='wmci')
parser.add_argument('--model_type', type=str, default='mlp')
parser.add_argument('--all', type=bool, default=True)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--model_conf_path', type=str, default='./grid_confs/conf')
parser.add_argument('--split', type=float, default=0.75)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--study_run', type=int, default=0)


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

# Get channels
rm_ch = set([])
args.rm_ch_str = ''
if args.remove_channels:
    rm_ch = set([int(i) for i in args.rm_ch.split('_')])
    args.rm_ch_str = args.rm_ch

channels = np.array([i for i in range(args.num_channels) if i not in rm_ch])

# Load data.
if args.rm_ch_str != '':
    data_path1 = args.data_path1 + '_' + args.rm_ch_str +'_notevt_' + str(args.contig_len) + '.pkl'
else:
    data_path1 = args.data_path1 +'_notevt_' + str(args.contig_len) + '.pkl'

if args.rm_ch_str != '':
    data_path2 = args.data_path2 + '_' + args.rm_ch_str +'_notevt_' + str(args.contig_len) + args.car_str + '.pkl'
else:
    data_path2 = args.data_path2 +'_notevt_' + str(args.contig_len) + args.car_str + '.pkl'

all_data = pickle.load(open(data_path1, 'rb'))
all_data.update(pickle.load(open(data_path2, 'rb')))

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

# file = open(f'{args.results_dir}/{args.study}_{args.lr}_{args.num_epochs}_{args.start_evt}_{args.filter_type}_{args.shuffle}_{args.include_evt}{balanced}.txt', 'a')

# Get clean train_test locations
train_pats, test_pats = get_train_test_pats_from_run(f'{args.results_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}.txt', args.study_run)
pats = {'train': train_pats, 'test': test_pats}

train_locs, test_locs = get_train_test_locs_from_pats(args, data, locs, pats)

train_pats = {args.study: set(), args.treat: set()}
test_pats = {args.study: set(), args.treat: set()}

for loc in train_locs:
    train_pats[loc[0]].add(loc[1])
for loc in test_locs:
    test_pats[loc[0]].add(loc[1])
print (train_pats)
print (test_pats)

args.num_psd = data[train_locs[0][0]][train_locs[0][1]][train_locs[0][2]].shape[1]

# For normalization
n = get_norms(args, data, channels, typs)

train_dataset = EEGSDataset(data_dict=data, locs=train_locs, study=args.study, treat=args.treat, norms=n)
test_dataset = EEGSDataset(data_dict=data, locs=test_locs, study=args.study, treat=args.treat, norms=n)

# Get Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Get Model
if args.model_type == 'mlp':
    model = MLP(num_channels=len(channels), num_psd=args.num_psd, out_dim=args.num_classes, conf=conf[1])
else:
    model = CNN(num_channels=len(channels), contig_len=args.contig_len, out_dim=args.num_classes, conf=conf[1])

if args.cuda:
    model = model.cuda()
print (model)

model.load_state_dict(torch.load(f'{args.weights_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{args.study_run}.pth'))

background = None

for X, y in train_dataloader:
    # select backgroud for shap
    if background is not None and background.shape[0] < 2000:
        background = torch.cat((background, X))
    elif background is None:
        background = X
    else:
        break

# DeepExplainer to explain predictions of the model
explainer = shap.DeepExplainer(model, background)

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

iter = 0
for X, y in tqdm(test_dataloader):
    logits = model(X)
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
                imps[iter * 32 + i] = shap_values[1][i]
                neg_imps[iter * 32 + i] = shap_values[0][i]
            else:
                imps[iter * 32 + i] = shap_values[1][i].sum(1)
                neg_imps[iter * 32 + i] = shap_values[0][i].sum(1)
        else:
            if args.all:
                imps[iter * 32 + i] = shap_values[0][i]
                neg_imps[iter * 32 + i] = shap_values[1][i]
            else:
                imps[iter * 32 + i] = shap_values[0][i].sum(1)
                neg_imps[iter * 32 + i] = shap_values[1][i].sum(1)

    iter += 1
    if iter == 100:
        break


pickle.dump(predictions, open(f'{args.results_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{args.study_run}_preds.pkl', 'wb'))
pickle.dump(targets, open(f'{args.results_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{args.study_run}_targets.pkl', 'wb'))
np.save(f'{args.results_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{args.study_run}_shap_imps', imps)
np.save(f'{args.results_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{args.study_run}_shap_negimps', neg_imps)

