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

from models import CNNS
from data import EEGSDataset
from utils import *

torch.cuda.set_device(1)


parser = argparse.ArgumentParser(description='Run EEG experiments')
parser.add_argument('--safe', type=bool, default=False)
parser.add_argument('--num_channels', type=int, default=19)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--remove_channels', type=str, default='14_15')
parser.add_argument('--meta_dir', type=str,default='meta_data_notevt')
parser.add_argument('--weights_dir', type=str,default='cnn_weights_spectral_notevt')
parser.add_argument('--results_dir', type=str,default='cnn_results_spectral_notevt')
parser.add_argument('--data_path', type=str,default='./mci_ctrl_ctrlb_psd_notevt.pkl')
parser.add_argument('--mask_path', type=str,default='./mask_notevt.npy')
parser.add_argument('--sf', type=float, default=250.)
parser.add_argument('--contig_len', type=int, default=250)
parser.add_argument('--art_thresh', type=int, default=0)
parser.add_argument('--lowcut', type=float, default=4.)
parser.add_argument('--highcut', type=float, default=40.)
parser.add_argument('--num_psd', type=int, default=26)
parser.add_argument('--min_contigs', type=int, default=5)
parser.add_argument('--balanced', type=bool, default=False)
parser.add_argument('--start_evt', type=bool, default=False)
parser.add_argument('--evt_starts', type=str, default='1_2')
parser.add_argument('--filter_type', type=str, default='global')
parser.add_argument('--include_evt', type=bool, default=False)
parser.add_argument('--shuffle', type=str, default='patient')
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--study', type=str, default='mci')
parser.add_argument('--treat', type=str, default='ctrlb')
parser.add_argument('--num_epochs', type=int, default=75)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--kernel', type=int, default=3)
parser.add_argument('--split', type=float, default=0.8)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--runs', type=int, default=20)

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

for run in range(args.runs):
    print (args.study, args.treat, args.shuffle, args.filter_type, args.start_evt, args.include_evt)

    balanced = ''
    if args.balanced:
        balanced = '_bal'

    safe_str = ''
    if args.safe:
        safe_str = '_safe'

    # Define class types
    typs = {args.study: 0, args.treat: 1}

    file = open(f'{args.results_dir}/{args.study}_{args.treat}_{args.remove_channels}_{args.start_evt}{safe_str}_oldnorm.txt', 'a')

    file.write(f'<<>>Run:{run}\n')

    # Get clean locations
    locs = get_clean_spectral_locs(args, all_data, typs, evts, channels)

    print (len(locs))
    file.write(f'Number of total thresholded contigs: {len(locs)}\n')

    # Get data info
    info = get_info(locs, args)

    # Print class wise patient info.
    print (info)
    file.write(f'Usable contigs: {str(info)}\n')


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
    train_locs, test_locs = get_spectral_train_test_locs(args, data, locs, balanced)

    train_pats = {args.study: set(), args.treat: set()}
    test_pats = {args.study: set(), args.treat: set()}

    for loc in train_locs:
        train_pats[loc[0]].add(loc[1])
    for loc in test_locs:
        test_pats[loc[0]].add(loc[1])
    print (train_pats)
    print (test_pats)
    file.write(f'Train patients: {str(train_pats)}\n')
    file.write(f'Test patients: {str(test_pats)}\n')

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

#     channel_maxs = np.full((len(channels), args.num_psd), -np.inf)
#     channel_mins = np.full((len(channels), args.num_psd), np.inf)
#     channel_means = np.zeros((len(channels), args.num_psd))
#     channel_stds = np.zeros((len(channels), args.num_psd))

#     count = 0

#     for typ in typs:
#         for pid in tqdm(range(len(data[typ]))):
#             if not len(data[typ][pid]):
#                 continue
#             channel_mins = np.minimum(np.min(data[typ][pid], 0), channel_mins)
#             channel_maxs = np.maximum(np.max(data[typ][pid], 0), channel_maxs)
#             channel_stds += np.std(data[typ][pid], 0)
#             channel_means += np.mean(data[typ][pid], 0)
#             count += 1
#     channel_means /= count
#     channel_stds /= count

#     norms = {'mins': torch.tensor(channel_mins).type(torch.float32), 'maxs': torch.tensor(channel_maxs).type(torch.float32)}


    train_batch_size = args.batch_size
    test_batch_size = 32
    n = None
    if args.normalize:
        n = norms

    train_dataset = EEGSDataset(data_dict=data, locs=train_locs, study=args.study, treat=args.treat, norms=n, evt=args.include_evt, mask=mask)
    test_dataset = EEGSDataset(data_dict=data, locs=test_locs, study=args.study, treat=args.treat, norms=n, evt=args.include_evt, mask=mask)

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    cnn = CNNS(num_channels=len(channels), num_psd=args.num_psd, kernel=args.kernel, out_dim=args.num_classes)

    if args.cuda:
        cnn = cnn.cuda()

    optimizer = optim.Adam(cnn.parameters(), args.lr)
    pbar = tqdm(range(args.num_epochs))

    for epoch in pbar:
        cnn.train()
        for X, y in train_dataloader:
            X = X.unsqueeze(1)
            logits = cnn(X)
            loss = nll(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description_str(desc='iter: '+str(epoch)+', loss: '+str(loss.item()))
        acc = 0
        tot = 0    
        cnn.eval()
        for X, y in train_dataloader:
            X = X.unsqueeze(1)
            logits = cnn(X)
            acc += mean_accuracy(logits, y)
            tot += logits.shape[0]
        print ('iter: '+str(epoch)+', acc: '+str(acc/tot))

        acc = 0
        tot = 0
        for X, y in test_dataloader:
            X = X.unsqueeze(1)
            logits = cnn(X)
            acc += mean_accuracy(logits, y)
            tot += logits.shape[0]

        print ('Test accuracy:', acc/tot)

    torch.save(cnn.state_dict(), f'{args.weights_dir}/{args.study}_{args.treat}_{args.remove_channels}_{args.start_evt}{safe_str}_{run}_split_oldnorm.pth')
#                         cnn = CNN(num_channels=len(channels) + int(args.include_evt)).cuda()
#                         cnn.load_state_dict(torch.load(f'{args.weights_dir}/{args.study}_{args.remove_channels}_{args.start_evt}_{args.contig_len}_{args.num_epochs}_{args.filter_type}_{args.shuffle}_{args.normalize}_{args.lr}.pth'))
    targets = None
    predictions = None
    for X, y in test_dataloader:
        X = X.unsqueeze(1)
        logits = cnn(X)
        if predictions is not None:
            predictions = torch.cat((predictions, logits.detach().cpu()))
        else:
            predictions = logits.detach().cpu()
        if targets is not None:
            targets = torch.cat((targets, y.cpu()))
        else:
            targets = y.cpu()

    file.write('CONTIG WISE METRICS\n\n')
    acc_macro = MulticlassAccuracy(average="macro", num_classes=2)
    acc_macro.update(predictions, targets)
    acc_none = MulticlassAccuracy(average=None, num_classes=2)
    acc_none.update(predictions, targets)
    acc = MulticlassAccuracy(num_classes=2)
    acc.update(predictions, targets)
    auroc = MulticlassAUROC(num_classes=2)
    auroc.update(predictions, targets)
    auprc = MulticlassAUPRC(num_classes=2)
    auprc.update(predictions, targets)
    recall_macro = MulticlassRecall(num_classes=2, average='macro')
    recall_macro.update(predictions, targets)
    recall_none = MulticlassRecall(num_classes=2, average=None)
    recall_none.update(predictions, targets)
    recall_weighted = MulticlassRecall(num_classes=2, average='weighted')
    recall_weighted.update(predictions, targets)
    recall = MulticlassRecall(num_classes=2)
    recall.update(predictions, targets)
    precision_macro = MulticlassPrecision(num_classes=2, average='macro')
    precision_macro.update(predictions, targets)
    precision_none = MulticlassPrecision(num_classes=2, average=None)
    precision_none.update(predictions, targets)
    precision_weighted = MulticlassPrecision(num_classes=2, average='weighted')
    precision_weighted.update(predictions, targets)
    precision = MulticlassPrecision(num_classes=2)
    precision.update(predictions, targets)
    f1_macro = MulticlassF1Score(num_classes=2, average='macro')
    f1_macro.update(predictions, targets)
    f1_none = MulticlassF1Score(num_classes=2, average=None)
    f1_none.update(predictions, targets)
    f1_weighted = MulticlassF1Score(num_classes=2, average='weighted')
    f1_weighted.update(predictions, targets)
    f1 = MulticlassF1Score(num_classes=2)
    f1.update(predictions, targets)
    confusion = MulticlassConfusionMatrix(num_classes=2)
    confusion.update(predictions, targets)

    print (f'Mean test accuracy: {acc.compute().item()}')
    file.write(f'Mean test accuracy: {acc.compute().item()}\n')
    print (f'Class mean test accuracy: {acc_macro.compute().item()}')
    file.write(f'Class mean test accuracy: {acc_macro.compute().item()}\n')
    acc_none_vals = acc_none.compute()
    print (f'Individual class test accuracies: "0" - {acc_none_vals[0].item()}, "1" - {acc_none_vals[1].item()}')
    file.write(f'Individual class test accuracies: "0" - {acc_none_vals[0].item()}, "1" - {acc_none_vals[1].item()}\n')
    print (f'Test AUPRC: {auprc.compute().item()}')
    file.write(f'Test AUPRC: {auprc.compute().item()}\n')
    print (f'Test AUROC: {auroc.compute().item()}')
    file.write(f'Test AUROC: {auroc.compute().item()}\n')
    print (f'Mean test recall: {recall.compute().item()}')
    file.write(f'Mean test recall: {recall.compute().item()}\n')
    print (f'Class mean test recall: {recall_macro.compute().item()}')
    file.write(f'Class mean test recall: {recall_macro.compute().item()}\n')
    print (f'Class weighted mean test recall: {recall_weighted.compute().item()}')
    file.write(f'Class weighted mean test recall: {recall_weighted.compute().item()}\n')
    recall_none_vals = recall_none.compute()
    print (f'Individual class test recalls: "0" - {recall_none_vals[0].item()}, "1" - {recall_none_vals[1].item()}')
    file.write(f'Individual class test recalls: "0" - {recall_none_vals[0].item()}, "1" - {recall_none_vals[1].item()}\n')
    print (f'Mean test precision: {precision.compute().item()}')
    file.write(f'Mean test precision: {precision.compute().item()}\n')
    print (f'Class mean test precision: {precision_macro.compute().item()}')
    file.write(f'Class mean test precision: {precision_macro.compute().item()}\n')
    print (f'Class weighted mean test precision: {precision_weighted.compute().item()}')
    file.write(f'Class weighted mean test precision: {precision_weighted.compute().item()}\n')
    precision_none_vals = precision_none.compute()
    print (f'Individual class test precisions: "0" - {precision_none_vals[0].item()}, "1" - {precision_none_vals[1].item()}')
    file.write(f'Individual class test precisions: "0" - {precision_none_vals[0].item()}, "1" - {precision_none_vals[1].item()}\n')
    print (f'Mean test f1: {f1.compute().item()}')
    file.write(f'Mean test f1: {f1.compute().item()}\n')
    print (f'Class mean test f1: {f1_macro.compute().item()}')
    file.write(f'Class mean test f1: {f1_macro.compute().item()}\n')
    print (f'Class weighted mean test f1: {f1_weighted.compute().item()}')
    file.write(f'Class weighted mean test f1: {f1_weighted.compute().item()}\n')
    f1_none_vals = f1_none.compute()
    print (f'Individual class test f1s: "0" - {f1_none_vals[0].item()}, "1" - {f1_none_vals[1].item()}')
    file.write(f'Individual class test f1s: "0" - {f1_none_vals[0].item()}, "1" - {f1_none_vals[1].item()}\n')
    confuse_vals = confusion.compute()
    print (f'Confusion matrix ("true_pred"): "0_0" - {int(confuse_vals[0][0].item())}, "0_1" - {int(confuse_vals[0][1].item())}, "1_0" - {int(confuse_vals[1][0].item())}, "1_1" - {int(confuse_vals[1][1].item())}')
    file.write(f'Confusion matrix ("true_pred"): "0_0" - {int(confuse_vals[0][0].item())}, "0_1" - {int(confuse_vals[0][1].item())}, "1_0" - {int(confuse_vals[1][0].item())}, "1_1" - {int(confuse_vals[1][1].item())}\n')

    acc = 0
    tot = 0

    for X, y in train_dataloader:    
        X = X.unsqueeze(1)
        logits = cnn(X)
        acc += mean_accuracy(logits, y)
        tot += logits.shape[0]
    print ('Train accuracy:', acc/tot)
    file.write(f'Mean train accuracy: {acc/tot}\n')

#                     temp_dataset = EEGSDataset(data_dict=data, locs=test_locs, study=args.study, treat=args.treat, norms=n, evt=args.include_evt, mask=mask)
    temp_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    count_0 = 0
    count_1 = 0
    count_pred_0 = 0
    count_pred_1 = 0
    prev_pid = 0

    accs = []

    pat_results = {
                args.study: {},
                args.treat: {}
                }

    acc = 0

    for i, (X, y) in enumerate(temp_loader):
        if test_locs[i][1] not in pat_results[test_locs[i][0]]:
            pat_results[test_locs[i][0]][test_locs[i][1]] = {'c': 0, 't': 0}
        X = X.unsqueeze(1)
        logits = cnn(X)
        acc = mean_accuracy(logits, y)
        pat_results[test_locs[i][0]][test_locs[i][1]]['t'] += 1
        if acc == 1:
            pat_results[test_locs[i][0]][test_locs[i][1]]['c'] += 1

    for typ in pat_results:
        for pat in pat_results[typ]:
            pat_results[typ][pat]['acc'] = pat_results[typ][pat]['c'] / pat_results[typ][pat]['t']

    print (pat_results)

    total_acc = 0
    total_count = 0
    pat_predictions = []
    pat_targets = []
    for typ in pat_results:
        typ_acc = 0
        typ_count = 0
        for pat in pat_results[typ]:
            true = typs[typ]
            other = 1 if true == 0 else 0
            pat_targets.append(true)
            pred = [0., 0.]
            if pat_results[typ][pat]['acc'] >= 0.5:
                typ_acc += 1
            pred[true] = pat_results[typ][pat]['acc']
            pred[other] = 1 - pat_results[typ][pat]['acc']
            pat_predictions.append(pred)
            typ_count += 1
        print (typ, typ_acc, typ_count, typ_acc / typ_count)
        file.write(f'cls: {typ}, correct: {typ_acc}, total: {typ_count}, acc: {typ_acc/typ_count}\n')
        total_acc += typ_acc
        total_count += typ_count
    print (total_acc, total_count, total_acc / total_count)

    predictions = torch.tensor(pat_predictions)
    targets = torch.tensor(pat_targets)

    file.write('PATIENT WISE METRICS\n\n')

    acc_macro = MulticlassAccuracy(average="macro", num_classes=2)
    acc_macro.update(predictions, targets)
    acc_none = MulticlassAccuracy(average=None, num_classes=2)
    acc_none.update(predictions, targets)
    acc = MulticlassAccuracy(num_classes=2)
    acc.update(predictions, targets)
    auroc = MulticlassAUROC(num_classes=2)
    auroc.update(predictions, targets)
    auprc = MulticlassAUPRC(num_classes=2)
    auprc.update(predictions, targets)
    recall_macro = MulticlassRecall(num_classes=2, average='macro')
    recall_macro.update(predictions, targets)
    recall_none = MulticlassRecall(num_classes=2, average=None)
    recall_none.update(predictions, targets)
    recall_weighted = MulticlassRecall(num_classes=2, average='weighted')
    recall_weighted.update(predictions, targets)
    recall = MulticlassRecall(num_classes=2)
    recall.update(predictions, targets)
    precision_macro = MulticlassPrecision(num_classes=2, average='macro')
    precision_macro.update(predictions, targets)
    precision_none = MulticlassPrecision(num_classes=2, average=None)
    precision_none.update(predictions, targets)
    precision_weighted = MulticlassPrecision(num_classes=2, average='weighted')
    precision_weighted.update(predictions, targets)
    precision = MulticlassPrecision(num_classes=2)
    precision.update(predictions, targets)
    f1_macro = MulticlassF1Score(num_classes=2, average='macro')
    f1_macro.update(predictions, targets)
    f1_none = MulticlassF1Score(num_classes=2, average=None)
    f1_none.update(predictions, targets)
    f1_weighted = MulticlassF1Score(num_classes=2, average='weighted')
    f1_weighted.update(predictions, targets)
    f1 = MulticlassF1Score(num_classes=2)
    f1.update(predictions, targets)
    confusion = MulticlassConfusionMatrix(num_classes=2)
    confusion.update(predictions, targets)

    print (f'Mean test accuracy: {acc.compute().item()}')
    file.write(f'Mean test accuracy: {acc.compute().item()}\n')
    print (f'Class mean test accuracy: {acc_macro.compute().item()}')
    file.write(f'Class mean test accuracy: {acc_macro.compute().item()}\n')
    acc_none_vals = acc_none.compute()
    print (f'Individual class test accuracies: "0" - {acc_none_vals[0].item()}, "1" - {acc_none_vals[1].item()}')
    file.write(f'Individual class test accuracies: "0" - {acc_none_vals[0].item()}, "1" - {acc_none_vals[1].item()}\n')
    print (f'Test AUPRC: {auprc.compute().item()}')
    file.write(f'Test AUPRC: {auprc.compute().item()}\n')
    print (f'Test AUROC: {auroc.compute().item()}')
    file.write(f'Test AUROC: {auroc.compute().item()}\n')
    print (f'Mean test recall: {recall.compute().item()}')
    file.write(f'Mean test recall: {recall.compute().item()}\n')
    print (f'Class mean test recall: {recall_macro.compute().item()}')
    file.write(f'Class mean test recall: {recall_macro.compute().item()}\n')
    print (f'Class weighted mean test recall: {recall_weighted.compute().item()}')
    file.write(f'Class weighted mean test recall: {recall_weighted.compute().item()}\n')
    recall_none_vals = recall_none.compute()
    print (f'Individual class test recalls: "0" - {recall_none_vals[0].item()}, "1" - {recall_none_vals[1].item()}')
    file.write(f'Individual class test recalls: "0" - {recall_none_vals[0].item()}, "1" - {recall_none_vals[1].item()}\n')
    print (f'Mean test precision: {precision.compute().item()}')
    file.write(f'Mean test precision: {precision.compute().item()}\n')
    print (f'Class mean test precision: {precision_macro.compute().item()}')
    file.write(f'Class mean test precision: {precision_macro.compute().item()}\n')
    print (f'Class weighted mean test precision: {precision_weighted.compute().item()}')
    file.write(f'Class weighted mean test precision: {precision_weighted.compute().item()}\n')
    precision_none_vals = precision_none.compute()
    print (f'Individual class test precisions: "0" - {precision_none_vals[0].item()}, "1" - {precision_none_vals[1].item()}')
    file.write(f'Individual class test precisions: "0" - {precision_none_vals[0].item()}, "1" - {precision_none_vals[1].item()}\n')
    print (f'Mean test f1: {f1.compute().item()}')
    file.write(f'Mean test f1: {f1.compute().item()}\n')
    print (f'Class mean test f1: {f1_macro.compute().item()}')
    file.write(f'Class mean test f1: {f1_macro.compute().item()}\n')
    print (f'Class weighted mean test f1: {f1_weighted.compute().item()}')
    file.write(f'Class weighted mean test f1: {f1_weighted.compute().item()}\n')
    f1_none_vals = f1_none.compute()
    print (f'Individual class test f1s: "0" - {f1_none_vals[0].item()}, "1" - {f1_none_vals[1].item()}')
    file.write(f'Individual class test f1s: "0" - {f1_none_vals[0].item()}, "1" - {f1_none_vals[1].item()}\n')
    confuse_vals = confusion.compute()
    print (f'Confusion matrix ("true_pred"): "0_0" - {int(confuse_vals[0][0].item())}, "0_1" - {int(confuse_vals[0][1].item())}, "1_0" - {int(confuse_vals[1][0].item())}, "1_1" - {int(confuse_vals[1][1].item())}')
    file.write(f'Confusion matrix ("true_pred"): "0_0" - {int(confuse_vals[0][0].item())}, "0_1" - {int(confuse_vals[0][1].item())}, "1_0" - {int(confuse_vals[1][0].item())}, "1_1" - {int(confuse_vals[1][1].item())}\n')

    file.write(f'All patient results: {str(pat_results)}\n')
