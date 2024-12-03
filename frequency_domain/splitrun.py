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
parser.add_argument('--data_path', type=str,default='./data/wmci_wctrl_psd')
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
parser.add_argument('--runs', type=int, default=30)


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

for run in range(args.runs):

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
    train_locs, test_locs = get_train_test_locs(args, data, locs)

    train_pats = {args.study: set(), args.treat: set()}
    test_pats = {args.study: set(), args.treat: set()}

    for loc in train_locs:
        train_pats[loc[0]].add(loc[1])
    for loc in test_locs:
        test_pats[loc[0]].add(loc[1])
    print (train_pats)
    print (test_pats)
    
    file = open(f'{args.results_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}{args.car_str}.txt', 'a')

    file.write(f'<<>>Run:{run}\n')
    file.write(f'Number of total thresholded contigs: {len(locs)}\n')
    file.write(f'Usable contigs: {str(info)}\n')
    file.write(f'Train patients: {str(train_pats)}\n')
    file.write(f'Test patients: {str(test_pats)}\n')

    args.num_psd = data[train_locs[0][0]][train_locs[0][1]][train_locs[0][2]].shape[1]

    
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

    optimizer = optim.Adam(model.parameters(), args.lr)
    pbar = tqdm(range(args.num_epochs))

    for epoch in pbar:
        model.train()
        for X, y in train_dataloader:
            logits = model(X)
            loss = nll(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description_str(desc='iter: '+str(epoch)+', loss: '+str(loss.item()))

        acc = 0
        tot = 0    
        model.eval()
        for X, y in train_dataloader:
            logits = model(X)
            acc += mean_accuracy(logits, y)
            tot += logits.shape[0]
        print ('train acc: '+str(acc/tot))
        file.write(f'epoch: {epoch}, train acc: {str((acc/tot).item())}, ')

        acc = 0
        tot = 0
        for X, y in test_dataloader:
            logits = model(X)
            acc += mean_accuracy(logits, y)
            tot += logits.shape[0]

        print ('test acc:', acc/tot)
        file.write(f'test acc: {str((acc/tot).item())}\n')
        
    torch.save(model.state_dict(), f'{args.weights_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}{args.car_str}_{run}.pth')

    targets = None
    predictions = None
    for X, y in test_dataloader:
        logits = model(X)
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
        logits = model(X)
        acc += mean_accuracy(logits, y)
        tot += logits.shape[0]
    print ('Train accuracy:', acc/tot)
    file.write(f'Mean train accuracy: {acc/tot}\n')

    file.write('PATIENT WISE METRICS\n\n')
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
        logits = model(X)
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
    file.close()