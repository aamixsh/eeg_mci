import numpy as np
import os
import argparse
import pickle
import shap


from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn, optim, autograd
from torch.utils.data import DataLoader

from torcheval.metrics import MulticlassAccuracy, MulticlassAUPRC, MulticlassAUROC, MulticlassConfusionMatrix, MulticlassF1Score, MulticlassPrecision, MulticlassRecall

from models import CNN
from data import EEGDataset
from utils import *


parser = argparse.ArgumentParser(description='Run EEG experiments')
parser.add_argument('--num_channels', type=int, default=19)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--remove_channels', type=bool, default=True)
parser.add_argument('--rm_ch', type=str, default='14_15')
parser.add_argument('--meta_dir', type=str,default='./meta_data')
parser.add_argument('--results_dir', type=str,default='./results_split')
parser.add_argument('--weights_dir', type=str,default='./weights_split')
parser.add_argument('--data_path', type=str,default='./data/mci_ctrlb')
parser.add_argument('--sf', type=float, default=250.)
parser.add_argument('--contig_len', type=int, default=250)
parser.add_argument('--art_thresh', type=int, default=0)
parser.add_argument('--lowcut', type=float, default=4.)
parser.add_argument('--highcut', type=float, default=40.)
parser.add_argument('--min_contigs', type=int, default=3)
parser.add_argument('--include_evt', type=bool, default=False)
parser.add_argument('--balanced', type=bool, default=True)
parser.add_argument('--norm_type', type=str, default='meanstd')
parser.add_argument('--study', type=str, default='mci')
parser.add_argument('--treat', type=str, default='ctrlb')
parser.add_argument('--model_type', type=str, default='cnn')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--model_conf_path', type=str, default='./grid_confs/conf')
parser.add_argument('--split', type=float, default=0.75)
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--total_runs', type=int, default=30)
parser.add_argument('--bg_num', type=int, default=1000)
parser.add_argument('--study_num', type=int, default=1500)
parser.add_argument('--all', type=bool, default=True)

args = parser.parse_args()

print('Args:')
for k, v in sorted(vars(args).items()):
  print("\t{}: {}".format(k, v))

# Set CUDA
if args.cuda:
    torch.cuda.set_device(0)

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

for run in range(args.runs):

    # Selects random training run to study.
    selected_run = 24
    print ('Selected_run:', selected_run)
    
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

    train_pats, test_pats = get_train_test_pats_from_run(f'{args.results_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}.txt', selected_run)
    pats = {'train': train_pats, 'test': test_pats}

    data = all_data
    # Get clean train_test locations
    train_locs, test_locs = get_train_test_locs_from_pats(args, data, locs, pats)

    train_pats = {args.study: set(), args.treat: set()}
    test_pats = {args.study: set(), args.treat: set()}

    for loc in train_locs:
        train_pats[loc[0]].add(loc[1])
    for loc in test_locs:
        test_pats[loc[0]].add(loc[1])
    print (train_pats)
    print (test_pats)
    
    # For normalization
    n = get_norms(args, data, channels, typs)

    train_dataset = EEGDataset(data_dict=data, locs=train_locs, study=args.study, treat=args.treat, norms=n, cuda=False)
    test_dataset = EEGDataset(data_dict=data, locs=test_locs, study=args.study, treat=args.treat, norms=n, cuda=False)

    # Get Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    # Get Model
    if args.model_type == 'mlp':
        model = MLP(num_channels=len(channels), num_psd=args.num_psd, out_dim=args.num_classes, conf=conf[1])
    else:
        model = CNN(num_channels=len(channels), contig_len=args.contig_len, out_dim=args.num_classes, conf=conf[1])

    # Load weights
    model.load_state_dict(torch.load(f'{args.weights_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{selected_run}.pth'))
    
    if args.cuda:
        model = model.cuda()
    print (model)
    
    background = None
    
    for X, y in train_dataloader:
        # select backgroud for shap
        if background is not None and background.shape[0] < args.bg_num:
            background = torch.cat((background, X))
        elif background is None:
            background = X
        else:
            break

    # DeepExplainer to explain predictions of the model
    explainer = shap.DeepExplainer(model, background)
    
    
    # Recreate train_dataloader without shuffle
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    
    sample_num = min(args.study_num, len(train_dataset))
    inds = np.random.choice(len(train_dataset), sample_num, replace=False)
    
    # Find patient_ids
    train_pat_ids = []
    for i in range(len(inds)):
        train_pat_ids.append(train_dataset.locs[inds[i]][:2])
    train_pat_ids = np.array(train_pat_ids)

    targets = None
    predictions = None

    imps = np.zeros((sample_num, train_dataset[0][0].shape[0]))
    neg_imps = np.zeros((sample_num, train_dataset[0][0].shape[0]))

    allt = ''
    if args.all:
        imps = np.zeros((sample_num, train_dataset[0][0].shape[0], train_dataset[0][0].shape[1]))
        neg_imps = np.zeros((sample_num, train_dataset[0][0].shape[0], train_dataset[0][0].shape[1]))
        allt = '_all'

#     file = open(f'{args.results_dir}/shap_imps{allt}_{args.study}_{args.treat}_{args.remove_channels}_{args.lr}_{args.num_epochs}_{args.start_evt}_{args.filter_type}_{args.shuffle}_{args.include_evt}{balanced}_{run}.txt', 'w')
#     filen = open(f'{args.results_dir}/shap_negimps{allt}_{args.study}_{args.treat}_{args.remove_channels}_{args.lr}_{args.num_epochs}_{args.start_evt}_{args.filter_type}_{args.shuffle}_{args.include_evt}{balanced}_{run}.txt', 'w')

    iter = 0
    for ind in tqdm(inds):
        X, y = train_dataset[ind]
        X = X.unsqueeze(0)
        y = y.unsqueeze(0)
        
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
        if args.all:
            imps[iter] = shap_values[1][0]
            neg_imps[iter] = shap_values[0][0]
        else:
            imps[iter] = shap_values[1][0].sum(1)
            neg_imps[iter] = shap_values[0][0].sum(1)
#         for i in range(len(X)):
#             if y[i]:
#                 if args.all:
#                     imps[iter * args.batch_size + i] = shap_values[1][i]
#                     neg_imps[iter * args.batch_size + i] = shap_values[0][i]
#                 else:
#                     imps[iter * args.batch_size + i] = shap_values[1][i].sum(1)
#                     neg_imps[iter * args.batch_size + i] = shap_values[0][i].sum(1)
#             else:
#                 if args.all:
#                     imps[iter * args.batch_size + i] = shap_values[0][i]
#                     neg_imps[iter * args.batch_size + i] = shap_values[1][i]
#                 else:
#                     imps[iter * args.batch_size + i] = shap_values[0][i].sum(1)
#                     neg_imps[iter * args.batch_size + i] = shap_values[1][i].sum(1)

#             for s in imps[iter * args.batch_size + i]:
#                 file.write(str(s)+',')
#             file.write('\n')
#             for s in neg_imps[iter * args.batch_size + i]:
#                 filen.write(str(s)+',')
#             filen.write('\n')
        iter += 1

    np.save(f'{args.results_dir}_{args.rm_ch_str}/train_shap_imps{allt}_{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{run}_fixed', imps)
    np.save(f'{args.results_dir}_{args.rm_ch_str}/train_shap_negimps{allt}_{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{run}_fixed', neg_imps)
    np.save(f'{args.results_dir}_{args.rm_ch_str}/train_preds{allt}_{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{run}_fixed', predictions.numpy())
    np.save(f'{args.results_dir}_{args.rm_ch_str}/train_targets{allt}_{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{run}_fixed', targets.numpy())
    np.save(f'{args.results_dir}_{args.rm_ch_str}/train_pat_ids{allt}_{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{run}_fixed', train_pat_ids)

    
    
    sample_num = min(args.study_num, len(test_dataset))
    inds = np.random.choice(len(test_dataset), sample_num, replace=False)
    
    # Find patient_ids
    test_pat_ids = []
    for i in range(len(test_dataset)):
        test_pat_ids.append(test_dataset.locs[i][:2])
    test_pat_ids = np.array(test_pat_ids)
    
    targets = None
    predictions = None

    imps = np.zeros((sample_num, test_dataset[0][0].shape[0]))
    neg_imps = np.zeros((sample_num, test_dataset[0][0].shape[0]))

    allt = ''
    if args.all:
        imps = np.zeros((sample_num, test_dataset[0][0].shape[0], test_dataset[0][0].shape[1]))
        neg_imps = np.zeros((sample_num, test_dataset[0][0].shape[0], test_dataset[0][0].shape[1]))
        allt = '_all'

#     file = open(f'{args.results_dir}/shap_imps{allt}_{args.study}_{args.treat}_{args.remove_channels}_{args.lr}_{args.num_epochs}_{args.start_evt}_{args.filter_type}_{args.shuffle}_{args.include_evt}{balanced}_{run}.txt', 'w')
#     filen = open(f'{args.results_dir}/shap_negimps{allt}_{args.study}_{args.treat}_{args.remove_channels}_{args.lr}_{args.num_epochs}_{args.start_evt}_{args.filter_type}_{args.shuffle}_{args.include_evt}{balanced}_{run}.txt', 'w')

    iter = 0
    for ind in tqdm(inds):
        X, y = test_dataset[ind]
        X = X.unsqueeze(0)
        y = y.unsqueeze(0)
        
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
        if args.all:
            imps[iter] = shap_values[1][0]
            neg_imps[iter] = shap_values[0][0]
        else:
            imps[iter] = shap_values[1][0].sum(1)
            neg_imps[iter] = shap_values[0][0].sum(1)

#         # compute shap values
#         shap_values = explainer.shap_values(X)
#         for i in range(len(X)):
#             if y[i]:
#                 if args.all:
#                     imps[iter * args.batch_size + i] = shap_values[1][i]
#                     neg_imps[iter * args.batch_size + i] = shap_values[0][i]
#                 else:
#                     imps[iter * args.batch_size + i] = shap_values[1][i].sum(1)
#                     neg_imps[iter * args.batch_size + i] = shap_values[0][i].sum(1)
#             else:
#                 if args.all:
#                     imps[iter * args.batch_size + i] = shap_values[0][i]
#                     neg_imps[iter * args.batch_size + i] = shap_values[1][i]
#                 else:
#                     imps[iter * args.batch_size + i] = shap_values[0][i].sum(1)
#                     neg_imps[iter * args.batch_size + i] = shap_values[1][i].sum(1)

#             for s in imps[iter * args.batch_size + i]:
#                 file.write(str(s)+',')
#             file.write('\n')
#             for s in neg_imps[iter * args.batch_size + i]:
#                 filen.write(str(s)+',')
#             filen.write('\n')
        iter += 1

    np.save(f'{args.results_dir}_{args.rm_ch_str}/test_shap_imps{allt}_{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{run}_fixed', imps)
    np.save(f'{args.results_dir}_{args.rm_ch_str}/test_shap_negimps{allt}_{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{run}_fixed', neg_imps)
    np.save(f'{args.results_dir}_{args.rm_ch_str}/test_preds{allt}_{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{run}_fixed', predictions.numpy())
    np.save(f'{args.results_dir}_{args.rm_ch_str}/test_targets{allt}_{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{run}_fixed', targets.numpy())
    np.save(f'{args.results_dir}_{args.rm_ch_str}/test_pat_ids{allt}_{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{run}_fixed', test_pat_ids)

#     print (torch.count_nonzero(predictions.argmax(1) == targets) / len(predictions), len(predictions))


#     temp_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#     count_0 = 0
#     count_1 = 0
#     count_pred_0 = 0
#     count_pred_1 = 0
#     prev_pid = 0

#     accs = []

#     pat_results = {
#                 args.study: {},
#                 args.treat: {}
#                 }

#     acc = 0

#     for i, (X, y) in enumerate(temp_loader):
#         if test_locs[i][1] not in pat_results[test_locs[i][0]]:
#             pat_results[test_locs[i][0]][test_locs[i][1]] = {'c': 0, 't': 0}
#         logits = cnn(X)
#         acc = mean_accuracy(logits, y)
#         pat_results[test_locs[i][0]][test_locs[i][1]]['t'] += 1
#         if acc == 1:
#             pat_results[test_locs[i][0]][test_locs[i][1]]['c'] += 1

#     for typ in pat_results:
#         for pat in pat_results[typ]:
#             pat_results[typ][pat]['acc'] = pat_results[typ][pat]['c'] / pat_results[typ][pat]['t']

#     print (pat_results)

#     total_acc = 0
#     total_count = 0
#     pat_predictions = []
#     pat_targets = []
#     for typ in pat_results:
#         typ_acc = 0
#         typ_count = 0
#         for pat in pat_results[typ]:
#             true = typs[typ]
#             other = 1 if true == 0 else 0
#             pat_targets.append(true)
#             pred = [0., 0.]
#             if pat_results[typ][pat]['acc'] >= 0.5:
#                 typ_acc += 1
#             pred[true] = pat_results[typ][pat]['acc']
#             pred[other] = 1 - pat_results[typ][pat]['acc']
#             pat_predictions.append(pred)
#             typ_count += 1
#         print (typ, typ_acc, typ_count, typ_acc / typ_count)
#         total_acc += typ_acc
#         total_count += typ_count
#     print (total_acc, total_count, total_acc / total_count)

#     predictions = torch.tensor(pat_predictions)
#     targets = torch.tensor(pat_targets)

#     print (torch.count_nonzero(predictions.argmax(1) == targets) / len(predictions), len(predictions))
