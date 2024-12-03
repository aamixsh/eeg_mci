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

torch.cuda.set_device(1)


parser = argparse.ArgumentParser(description='Run EEG experiments')
parser.add_argument('--num_channels', type=int, default=19)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--remove_channels', type=str, default='14_15')
parser.add_argument('--meta_dir', type=str,default='meta_data')
parser.add_argument('--weights_dir', type=str,default='weights')
parser.add_argument('--results_dir', type=str,default='results_kfold')
parser.add_argument('--data_path', type=str,default='./all_data_latest.pkl')
parser.add_argument('--sf', type=float, default=250.)
parser.add_argument('--contig_len', type=int, default=250)
parser.add_argument('--art_thresh', type=int, default=0)
parser.add_argument('--lowcut', type=float, default=4.)
parser.add_argument('--highcut', type=float, default=40.)
parser.add_argument('--min_contigs', type=int, default=5)
parser.add_argument('--start_evt', type=bool, default=True)
parser.add_argument('--evt_starts', type=str, default='1_2')
parser.add_argument('--filter_type', type=str, default='global')
parser.add_argument('--include_evt', type=bool, default=False)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--balanced', type=bool, default=True)
parser.add_argument('--study', type=str, default='ctrl')
parser.add_argument('--treat', type=str, default='rb')
parser.add_argument('--num_epochs', type=int, default=75)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--kernel', type=int, default=8)
parser.add_argument('--depth', type=int, default=3)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--runs', type=int, default=3)

args = parser.parse_args()

print('Args:')
for k, v in sorted(vars(args).items()):
  print("\t{}: {}".format(k, v))

if args.cuda:
    torch.cuda.set_device(1)

# Load data.
all_data = pickle.load(open(args.data_path, 'rb'))

# Get channels
if args.remove_channels == '':
    rm_ch = set()
else:
    rm_ch = set([int(i) for i in args.remove_channels.split('_')])
channels = np.array([i for i in range(args.num_channels) if i not in rm_ch])

# EVT start vals
evts = [int(i) for i in args.evt_starts.split('_')]

for run in range(args.runs):
    start_evts = [True]
    filter_types = ['global']
    bals = [False]
    include_evts = [False]
    for include_evt in include_evts:
        for start_evt in start_evts:
            for filter_type in filter_types:
                for bal in bals:
                    args.balanced = bal
                    args.filter_type = filter_type
                    args.start_evt = start_evt
                    args.include_evt = include_evt

                    print (args.study, args.treat, args.filter_type, args.start_evt, args.include_evt)

                    if args.balanced:
                        balanced = '_bal'
                    else:
                        balanced = ''

                    # Define class types
                    typs = {args.study: 0, args.treat: 1}

                    file = open(f'{args.results_dir}/{args.study}_{args.treat}_{args.start_evt}_{args.filter_type}_{args.include_evt}{balanced}.txt', 'w')

                    file.write(f'<<>>Run:{run}\n')
                    
                    # Get clean locations
                    locs = get_clean_locs(args, all_data, typs, evts, channels)

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

                    # Get filtered data.
                    new_data = all_data
                    if args.filter_type == 'global':
                        for typ in typs:
                            for pid, datum in tqdm(enumerate(all_data[typ]['eeg'])):
                                new_datum = []
                                for c in range(all_data[typ]['eeg'][pid].shape[1]):
#                                     new_datum_c = butter_bandpass_filter(datum[:, c], args.lowcut, args.highcut, args.sf, order=10)
                                    new_datum.append(datum[:, c])
                                new_data[typ]['eeg'][pid] = np.array(new_datum).T

                    # filtered data dict
                    data = {args.study: 
                                    {
                                    'eeg': [],
                                    'evt': []
                                    }, 
                            args.treat: 
                                    {
                                    'eeg': [],
                                    'evt': []
                                    }
                           }

                    new_locs = []

                    prev_pid = 0
                    prev_loc = None
                    contig_ind = 0
                    for loc in tqdm(locs):
                        typ, pid, ind = loc
                        if pid in pats[typ]:
                            while len(data[typ]['eeg']) <= pid:
                                data[typ]['eeg'].append([])
                                data[typ]['evt'].append([])
                                contig_ind = 0
                            new_datum = new_data[typ]['eeg'][pid][ind: ind + args.contig_len, channels]
                            evt = new_data[typ]['evt'][pid][ind: ind + args.contig_len]
                            if args.filter_type == 'local': 
                                nnew_datum = []
                                for c in range(new_datum.shape[1]):
                                    nnew_datum_c = butter_bandpass_filter(new_datum[:, c], args.lowcut, args.highcut, args.sf, order=10)
                                    nnew_datum.append(nnew_datum_c)
                                new_datum = np.array(nnew_datum).T
                    #             new_datum = butter_bandpass_filter(new_datum, lowcut, highcut, sf, order=10)
                            data[typ]['eeg'][pid].append(new_datum)
                            data[typ]['evt'][pid].append(evt)
                            new_locs.append((typ, pid, contig_ind))
                            contig_ind += 1

                    for typ in data:
                        for pid in range(len(data[typ]['eeg'])):
                            data[typ]['eeg'][pid] = np.array(data[typ]['eeg'][pid])
                            data[typ]['evt'][pid] = np.array(data[typ]['evt'][pid])
                    # for typ in typs:
                    #     for i in range(len(data[typ]['eeg'])):
                    #         print (typ, data[typ]['eeg'][i].shape, data[typ]['evt'][i].shape)
                    
                    
                    # For normalization
                    channel_maxs = np.full((len(channels)), -np.inf)
                    channel_mins = np.full((len(channels)), np.inf)
                    channel_means = np.zeros(len(channels))
                    channel_stds = np.zeros(len(channels))

                    count = 0

                    for typ in data:
                        for pid in tqdm(range(len(data[typ]['eeg']))):
                            if not len(data[typ]['eeg'][pid]):
                                continue
                            for i in range(len(channels)):
                                pmin = np.min(data[typ]['eeg'][pid][:, :, i])
                                pmax = np.max(data[typ]['eeg'][pid][:, :, i])
                                pstd = np.std(data[typ]['eeg'][pid][:, :, i])
                                pmean = np.mean(data[typ]['eeg'][pid][:, :, i])
                                if pmin < channel_mins[i]:
                                    channel_mins[i] = pmin
                                if pmax > channel_maxs[i]:
                                    channel_maxs[i] = pmax
                                channel_means[i] += pmean
                                channel_stds[i] += pstd
                        count += len(data[typ]['eeg'])
                    channel_means /= count
                    channel_stds /= count

                    print (channel_maxs)
                    print (channel_mins)
                    # print (channel_means)
                    # print (channel_stds)
                    norms = {'mean': torch.tensor(channel_means).type(torch.float32), 'std': torch.tensor(channel_stds).type(torch.float32)}
                    # norms = {'mean': channel_means, 'std': channel_stds}
                    print (norms)


                    # Run experiment for each patient as test.

                    mean_test_accuracy = 0.
                    mean_pat_test_accuracy = 0.
                    folds = 0
                    
                    for typ_g in data:
                        for pat_id, pat in enumerate(data[typ_g]['eeg']):
                            test_patient = (typ_g, pat_id)
                            train_locs = []
                            test_locs = []
                            class_labels = []
                            for loc in new_locs:
                                if loc[0] == typ_g and loc[1] == pat_id:
                                    test_locs.append(loc)
                                else:
                                    train_locs.append(loc)
                                    if loc[0] == args.study:
                                        class_labels.append(0)
                                    else:
                                        class_labels.append(1)
                                        
                            print (test_patient)
                            print (test_locs)
#                             if not test_locs:
#                                 continue
                                
                            file.write(f'Fold: {folds}\n\n')

                            class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)).type(torch.float32).cuda()
                            if not args.balanced:
                                class_weights = None
                            print ('class_weights:', class_weights)

                            train_pats = {args.study: set(), args.treat: set()}
                            test_pats = {args.study: set(), args.treat: set()}

                            for loc in train_locs:
                                train_pats[loc[0]].add(loc[1])
                            for loc in test_locs:
                                test_pats[loc[0]].add(loc[1])

                            print (train_pats)
                            print (test_pats)
                            

                            file.write(f'Train patients: {str(train_pats)}\n')
                            file.write(f'Test patient: {str(test_pats)}\n')


                            train_batch_size = args.batch_size
                            test_batch_size = 32
                            n = None
                            if args.normalize:
                                n = norms

                            train_dataset = EEGDataset(data_dict=data, locs=train_locs, study=args.study, treat=args.treat, norms=n, evt=args.include_evt)
                            test_dataset = EEGDataset(data_dict=data, locs=test_locs, study=args.study, treat=args.treat, norms=n, evt=args.include_evt)

                            train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
                            test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


                            if len(channels) == 3:
                                cnn = CNNs(num_channels=len(channels) + int(args.include_evt), kernel=args.kernel, out_dim=args.num_classes)
                            else:
                                cnn = CNN(num_channels=len(channels) + int(args.include_evt), kernel=args.kernel, out_dim=args.num_classes)
                            if args.cuda:
                                cnn = cnn.cuda()

                            optimizer = optim.Adam(cnn.parameters(), args.lr)
                            pbar = tqdm(range(args.num_epochs))

                            for epoch in pbar:
                                cnn.train()
                                for X, y in train_dataloader:
                                    logits = cnn(X)
                                    loss = nll(logits, y, weight=class_weights)

                                    optimizer.zero_grad()
                                    loss.backward()
                                    optimizer.step()

                                    pbar.set_description_str(desc='iter: '+str(epoch)+', loss: '+str(loss.item()))
                                acc = 0
                                tot = 0    
                                cnn.eval()
                                for X, y in train_dataloader:
                                    logits = cnn(X)
                                    acc += mean_accuracy(logits, y)
                                    tot += logits.shape[0]
                                print ('iter: '+str(epoch)+', acc: '+str(acc/tot))

                                acc = 0
                                tot = 0
                                for X, y in test_dataloader:
                                    logits = cnn(X)
                                    acc += mean_accuracy(logits, y)
                                    tot += logits.shape[0]

                                print ('Test accuracy:', acc/tot)

    #                         torch.save(cnn.state_dict(), f'{args.weights_dir}/{args.study}_{args.remove_channels}_{args.start_evt}_{args.contig_len}_{args.num_epochs}_{args.filter_type}_{args.shuffle}_{args.normalize}_{args.lr}.pth')
            #                         cnn = CNN(num_channels=len(channels) + int(args.include_evt)).cuda()
            #                         cnn.load_state_dict(torch.load(f'{args.weights_dir}/{args.study}_{args.remove_channels}_{args.start_evt}_{args.contig_len}_{args.num_epochs}_{args.filter_type}_{args.shuffle}_{args.normalize}_{args.lr}.pth'))
                            targets = None
                            predictions = None
                            for X, y in test_dataloader:
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
#                             acc_macro = MulticlassAccuracy(average="macro", num_classes=2)
#                             acc_macro.update(predictions, targets)
#                             acc_none = MulticlassAccuracy(average=None, num_classes=2)
#                             acc_none.update(predictions, targets)
                            acc = MulticlassAccuracy(num_classes=2)
                            acc.update(predictions, targets)
#                             auroc = MulticlassAUROC(num_classes=2)
#                             auroc.update(predictions, targets)
#                             auprc = MulticlassAUPRC(num_classes=2)
#                             auprc.update(predictions, targets)
#                             recall_macro = MulticlassRecall(num_classes=2, average='macro')
#                             recall_macro.update(predictions, targets)
#                             recall_none = MulticlassRecall(num_classes=2, average=None)
#                             recall_none.update(predictions, targets)
#                             recall_weighted = MulticlassRecall(num_classes=2, average='weighted')
#                             recall_weighted.update(predictions, targets)
#                             recall = MulticlassRecall(num_classes=2)
#                             recall.update(predictions, targets)
#                             precision_macro = MulticlassPrecision(num_classes=2, average='macro')
#                             precision_macro.update(predictions, targets)
#                             precision_none = MulticlassPrecision(num_classes=2, average=None)
#                             precision_none.update(predictions, targets)
#                             precision_weighted = MulticlassPrecision(num_classes=2, average='weighted')
#                             precision_weighted.update(predictions, targets)
#                             precision = MulticlassPrecision(num_classes=2)
#                             precision.update(predictions, targets)
#                             f1_macro = MulticlassF1Score(num_classes=2, average='macro')
#                             f1_macro.update(predictions, targets)
#                             f1_none = MulticlassF1Score(num_classes=2, average=None)
#                             f1_none.update(predictions, targets)
#                             f1_weighted = MulticlassF1Score(num_classes=2, average='weighted')
#                             f1_weighted.update(predictions, targets)
#                             f1 = MulticlassF1Score(num_classes=2)
#                             f1.update(predictions, targets)
#                             confusion = MulticlassConfusionMatrix(num_classes=2)
#                             confusion.update(predictions, targets)

                            print (f'Mean test accuracy: {acc.compute().item()}')
                            file.write(f'Mean test accuracy: {acc.compute().item()}\n')
                            mean_test_accuracy += acc.compute().item()
#                             print (f'Class mean test accuracy: {acc_macro.compute().item()}')
#                             file.write(f'Class mean test accuracy: {acc_macro.compute().item()}\n')
#                             acc_none_vals = acc_none.compute()
#                             print (f'Individual class test accuracies: "0" - {acc_none_vals[0].item()}, "1" - {acc_none_vals[1].item()}')
#                             file.write(f'Individual class test accuracies: "0" - {acc_none_vals[0].item()}, "1" - {acc_none_vals[1].item()}\n')
#                             print (f'Test AUPRC: {auprc.compute().item()}')
#                             file.write(f'Test AUPRC: {auprc.compute().item()}\n')
#                             print (f'Test AUROC: {auroc.compute().item()}')
#                             file.write(f'Test AUROC: {auroc.compute().item()}\n')
#                             print (f'Mean test recall: {recall.compute().item()}')
#                             file.write(f'Mean test recall: {recall.compute().item()}\n')
#                             print (f'Class mean test recall: {recall_macro.compute().item()}')
#                             file.write(f'Class mean test recall: {recall_macro.compute().item()}\n')
#                             print (f'Class weighted mean test recall: {recall_weighted.compute().item()}')
#                             file.write(f'Class weighted mean test recall: {recall_weighted.compute().item()}\n')
#                             recall_none_vals = recall_none.compute()
#                             print (f'Individual class test recalls: "0" - {recall_none_vals[0].item()}, "1" - {recall_none_vals[1].item()}')
#                             file.write(f'Individual class test recalls: "0" - {recall_none_vals[0].item()}, "1" - {recall_none_vals[1].item()}\n')
#                             print (f'Mean test precision: {precision.compute().item()}')
#                             file.write(f'Mean test precision: {precision.compute().item()}\n')
#                             print (f'Class mean test precision: {precision_macro.compute().item()}')
#                             file.write(f'Class mean test precision: {precision_macro.compute().item()}\n')
#                             print (f'Class weighted mean test precision: {precision_weighted.compute().item()}')
#                             file.write(f'Class weighted mean test precision: {precision_weighted.compute().item()}\n')
#                             precision_none_vals = precision_none.compute()
#                             print (f'Individual class test precisions: "0" - {precision_none_vals[0].item()}, "1" - {precision_none_vals[1].item()}')
#                             file.write(f'Individual class test precisions: "0" - {precision_none_vals[0].item()}, "1" - {precision_none_vals[1].item()}\n')
#                             print (f'Mean test f1: {f1.compute().item()}')
#                             file.write(f'Mean test f1: {f1.compute().item()}\n')
#                             print (f'Class mean test f1: {f1_macro.compute().item()}')
#                             file.write(f'Class mean test f1: {f1_macro.compute().item()}\n')
#                             print (f'Class weighted mean test f1: {f1_weighted.compute().item()}')
#                             file.write(f'Class weighted mean test f1: {f1_weighted.compute().item()}\n')
#                             f1_none_vals = f1_none.compute()
#                             print (f'Individual class test f1s: "0" - {f1_none_vals[0].item()}, "1" - {f1_none_vals[1].item()}')
#                             file.write(f'Individual class test f1s: "0" - {f1_none_vals[0].item()}, "1" - {f1_none_vals[1].item()}\n')
#                             confuse_vals = confusion.compute()
#                             print (f'Confusion matrix ("true_pred"): "0_0" - {int(confuse_vals[0][0].item())}, "0_1" - {int(confuse_vals[0][1].item())}, "1_0" - {int(confuse_vals[1][0].item())}, "1_1" - {int(confuse_vals[1][1].item())}')
#                             file.write(f'Confusion matrix ("true_pred"): "0_0" - {int(confuse_vals[0][0].item())}, "0_1" - {int(confuse_vals[0][1].item())}, "1_0" - {int(confuse_vals[1][0].item())}, "1_1" - {int(confuse_vals[1][1].item())}\n')

                            acc = 0
                            tot = 0

                            for X, y in train_dataloader:    
                                logits = cnn(X)
                                acc += mean_accuracy(logits, y)
                                tot += logits.shape[0]
                            print ('Train accuracy:', acc/tot)
                            file.write(f'Mean train accuracy: {acc/tot}\n')

                            temp_dataset = EEGDataset(data_dict=data, locs=test_locs, study=args.study, treat=args.treat, norms=n, evt=args.include_evt)
                            temp_loader = DataLoader(temp_dataset, batch_size=1, shuffle=False)

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
                                if typ_count:
                                    print (typ, typ_acc, typ_count, typ_acc / typ_count)
                                    file.write(f'cls: {typ}, correct: {typ_acc}, total: {typ_count}, acc: {typ_acc/typ_count}\n')
                                total_acc += typ_acc
                                total_count += typ_count
                            if total_count:
                                print (total_acc, total_count, total_acc / total_count)

                            predictions = torch.tensor(pat_predictions)
                            targets = torch.tensor(pat_targets)

                            file.write('PATIENT WISE METRICS\n\n')

#                             acc_macro = MulticlassAccuracy(average="macro", num_classes=2)
#                             acc_macro.update(predictions, targets)
#                             acc_none = MulticlassAccuracy(average=None, num_classes=2)
#                             acc_none.update(predictions, targets)
                            acc = MulticlassAccuracy(num_classes=2)
                            acc.update(predictions, targets)
#                             auroc = MulticlassAUROC(num_classes=2)
#                             auroc.update(predictions, targets)
#                             auprc = MulticlassAUPRC(num_classes=2)
#                             auprc.update(predictions, targets)
#                             recall_macro = MulticlassRecall(num_classes=2, average='macro')
#                             recall_macro.update(predictions, targets)
#                             recall_none = MulticlassRecall(num_classes=2, average=None)
#                             recall_none.update(predictions, targets)
#                             recall_weighted = MulticlassRecall(num_classes=2, average='weighted')
#                             recall_weighted.update(predictions, targets)
#                             recall = MulticlassRecall(num_classes=2)
#                             recall.update(predictions, targets)
#                             precision_macro = MulticlassPrecision(num_classes=2, average='macro')
#                             precision_macro.update(predictions, targets)
#                             precision_none = MulticlassPrecision(num_classes=2, average=None)
#                             precision_none.update(predictions, targets)
#                             precision_weighted = MulticlassPrecision(num_classes=2, average='weighted')
#                             precision_weighted.update(predictions, targets)
#                             precision = MulticlassPrecision(num_classes=2)
#                             precision.update(predictions, targets)
#                             f1_macro = MulticlassF1Score(num_classes=2, average='macro')
#                             f1_macro.update(predictions, targets)
#                             f1_none = MulticlassF1Score(num_classes=2, average=None)
#                             f1_none.update(predictions, targets)
#                             f1_weighted = MulticlassF1Score(num_classes=2, average='weighted')
#                             f1_weighted.update(predictions, targets)
#                             f1 = MulticlassF1Score(num_classes=2)
#                             f1.update(predictions, targets)
#                             confusion = MulticlassConfusionMatrix(num_classes=2)
#                             confusion.update(predictions, targets)

                            print (f'Mean test accuracy: {acc.compute().item()}')
                            file.write(f'Mean test accuracy: {acc.compute().item()}\n')
                            mean_pat_test_accuracy += acc.compute().item()
#                             print (f'Class mean test accuracy: {acc_macro.compute().item()}')
#                             file.write(f'Class mean test accuracy: {acc_macro.compute().item()}\n')
#                             acc_none_vals = acc_none.compute()
#                             print (f'Individual class test accuracies: "0" - {acc_none_vals[0].item()}, "1" - {acc_none_vals[1].item()}')
#                             file.write(f'Individual class test accuracies: "0" - {acc_none_vals[0].item()}, "1" - {acc_none_vals[1].item()}\n')
#                             print (f'Test AUPRC: {auprc.compute().item()}')
#                             file.write(f'Test AUPRC: {auprc.compute().item()}\n')
#                             print (f'Test AUROC: {auroc.compute().item()}')
#                             file.write(f'Test AUROC: {auroc.compute().item()}\n')
#                             print (f'Mean test recall: {recall.compute().item()}')
#                             file.write(f'Mean test recall: {recall.compute().item()}\n')
#                             print (f'Class mean test recall: {recall_macro.compute().item()}')
#                             file.write(f'Class mean test recall: {recall_macro.compute().item()}\n')
#                             print (f'Class weighted mean test recall: {recall_weighted.compute().item()}')
#                             file.write(f'Class weighted mean test recall: {recall_weighted.compute().item()}\n')
#                             recall_none_vals = recall_none.compute()
#                             print (f'Individual class test recalls: "0" - {recall_none_vals[0].item()}, "1" - {recall_none_vals[1].item()}')
#                             file.write(f'Individual class test recalls: "0" - {recall_none_vals[0].item()}, "1" - {recall_none_vals[1].item()}\n')
#                             print (f'Mean test precision: {precision.compute().item()}')
#                             file.write(f'Mean test precision: {precision.compute().item()}\n')
#                             print (f'Class mean test precision: {precision_macro.compute().item()}')
#                             file.write(f'Class mean test precision: {precision_macro.compute().item()}\n')
#                             print (f'Class weighted mean test precision: {precision_weighted.compute().item()}')
#                             file.write(f'Class weighted mean test precision: {precision_weighted.compute().item()}\n')
#                             precision_none_vals = precision_none.compute()
#                             print (f'Individual class test precisions: "0" - {precision_none_vals[0].item()}, "1" - {precision_none_vals[1].item()}')
#                             file.write(f'Individual class test precisions: "0" - {precision_none_vals[0].item()}, "1" - {precision_none_vals[1].item()}\n')
#                             print (f'Mean test f1: {f1.compute().item()}')
#                             file.write(f'Mean test f1: {f1.compute().item()}\n')
#                             print (f'Class mean test f1: {f1_macro.compute().item()}')
#                             file.write(f'Class mean test f1: {f1_macro.compute().item()}\n')
#                             print (f'Class weighted mean test f1: {f1_weighted.compute().item()}')
#                             file.write(f'Class weighted mean test f1: {f1_weighted.compute().item()}\n')
#                             f1_none_vals = f1_none.compute()
#                             print (f'Individual class test f1s: "0" - {f1_none_vals[0].item()}, "1" - {f1_none_vals[1].item()}')
#                             file.write(f'Individual class test f1s: "0" - {f1_none_vals[0].item()}, "1" - {f1_none_vals[1].item()}\n')
#                             confuse_vals = confusion.compute()
#                             print (f'Confusion matrix ("true_pred"): "0_0" - {int(confuse_vals[0][0].item())}, "0_1" - {int(confuse_vals[0][1].item())}, "1_0" - {int(confuse_vals[1][0].item())}, "1_1" - {int(confuse_vals[1][1].item())}')
#                             file.write(f'Confusion matrix ("true_pred"): "0_0" - {int(confuse_vals[0][0].item())}, "0_1" - {int(confuse_vals[0][1].item())}, "1_0" - {int(confuse_vals[1][0].item())}, "1_1" - {int(confuse_vals[1][1].item())}\n')

                            file.write(f'All patient results: {str(pat_results)}\n')
                            
                            folds += 1
                    
                    print (f'Average mean test accuracy (contig-wise): {mean_test_accuracy / folds}')
                    print (f'Average mean test accuracy (patient-wise): {mean_pat_test_accuracy / folds}')
                    file.write(f'Average mean test accuracy (contig-wise): {mean_test_accuracy / folds}\n')                    
                    file.write(f'Average mean test accuracy (patient-wise): {mean_pat_test_accuracy / folds}\n')                    
