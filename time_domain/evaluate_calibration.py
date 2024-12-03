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

from models import CNN
from data import EEGDataset
from utils import *

from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist

parser = argparse.ArgumentParser(description='Run EEG Spectral experiments')
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
parser.add_argument('--gmm_components', type=int, default=5)
parser.add_argument('--norm_type', type=str, default='meanstd')
parser.add_argument('--study', type=str, default='mci')
parser.add_argument('--treat', type=str, default='ctrlb')
parser.add_argument('--model_type', type=str, default='cnn')
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
    torch.cuda.set_device(0)

# Get channels
rm_ch = set([])
args.rm_ch_str = ''
if args.remove_channels:
    rm_ch = set([int(i) for i in args.rm_ch.split('_')])
    args.rm_ch_str = args.rm_ch

channels = np.array([i for i in range(args.num_channels) if i not in rm_ch])


# contig_lens = ['250', '500', '1000', '2000']
contig_lens = ['250']
# norms = ['none', 'minmax', 'meanstd']
norms = ['meanstd']
# balances = ['True', 'False']
balances = ['False']

for contig_len in contig_lens:
    
    args.contig_len = contig_len
    # Load data.
    if args.rm_ch_str != '':
        data_path = args.data_path + '_' + args.rm_ch_str +'_notevt_' + str(args.contig_len) + '.pkl'
    else:
        data_path = args.data_path +'_notevt_' + str(args.contig_len) + '.pkl'
    all_data = pickle.load(open(data_path, 'rb'))
    
    for norm in norms:
        for balance in balances:
            args.norm_type = norm
            args.balanced = balance

            # Define class types
            typs = {args.study: 0, args.treat: 1}

            conf_path = args.model_conf_path + '_' + args.model_type + '_' + str(args.contig_len) + '_' + str(len(channels)) + '_best.txt'

            conf = read_grid_confs(conf_path)[0]

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
            file = open(f'results_split_14_15/mci_ctrlb_{contig_len}_{norm}_{balance}_calibration.txt', 'a')
            for run in range(args.runs):

                train_pats, test_pats = get_train_test_pats_from_run(f'{args.results_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}.txt', run)
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

                args.num_psd = data[train_locs[0][0]][train_locs[0][1]][train_locs[0][2]].shape[1]


                n = get_norms(args, data, channels, typs)

                train_dataset = EEGDataset(data_dict=data, locs=train_locs, study=args.study, treat=args.treat, norms=n)
                test_dataset = EEGDataset(data_dict=data, locs=test_locs, study=args.study, treat=args.treat, norms=n)

                # Get Dataloaders
                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
#                 # For GMM
#                 trainX, trainy, testX, testy = None, None, None, None

#                 for X, y in tqdm(train_dataset):
#                     if trainX is None:
#                         trainX, trainy = X.unsqueeze(0).cpu().numpy(), y.unsqueeze(0).cpu().numpy()
#                     else:
#                         trainX, trainy = np.concatenate((trainX, X.unsqueeze(0).cpu().numpy())), np.concatenate((trainy, y.unsqueeze(0).cpu().numpy()))

#                 for X, y in tqdm(test_dataset):
#                     if testX is None:
#                         testX, testy = X.unsqueeze(0).cpu().numpy(), y.unsqueeze(0).cpu().numpy()
#                     else:
#                         testX, testy = np.concatenate((testX, X.unsqueeze(0).cpu().numpy())), np.concatenate((testy, y.unsqueeze(0).cpu().numpy()))
                        
#                 trainX = trainX.reshape(-1, trainX.shape[1] * trainX.shape[2])
#                 testX = testX.reshape(-1, testX.shape[1] * testX.shape[2])
                
#                 class_0_inds = np.where(trainy == 0)[0]
#                 trainX0 = trainX[class_0_inds]

#                 class_1_inds = np.where(trainy == 1)[0]
#                 trainX1 = trainX[class_1_inds]
                
#                 gm0 = GaussianMixture(n_components=args.gmm_components, random_state=0, verbose=1).fit(trainX0)
#                 gm1 = GaussianMixture(n_components=args.gmm_components, random_state=0, verbose=1).fit(trainX1)
                
#                 differences = cdist(gm0.means_, gm1.means_)
#                 print (differences)

#                 weights = differences / np.max(differences)
#                 print (weights)
                
#                 pred0, pred1 = gm0.predict(testX), gm1.predict(testX)
#                 test_weights = weights[pred0, pred1]
                
                
                # Get Model
                if args.model_type == 'mlp':
                    model = MLP(num_channels=len(channels), num_psd=args.num_psd, out_dim=args.num_classes, conf=conf[1])
                else:
                    model = CNN(num_channels=len(channels), contig_len=args.contig_len, out_dim=args.num_classes, conf=conf[1])

                # Load weights
                model.load_state_dict(torch.load(f'{args.weights_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{run}.pth'))


                if args.cuda:
                    model = model.cuda()
                print (model)

                model.eval()
                
                cal = 0
                tot = 0
                
                for X, y in test_dataloader:    
                    logits = model(X)
                    cal += mce(logits, y, 2) * logits.shape[0]
                    tot += logits.shape[0]
                    
                file.write(f'{cal/tot}\n')
                
                
#                 file.write(f'<<>>Run:{run}\n')
#                 file.write('PATIENT WISE METRICS\n\n')
#                 temp_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#                 count_0 = 0
#                 count_1 = 0
#                 count_pred_0 = 0
#                 count_pred_1 = 0
#                 prev_pid = 0

#                 accs = []

#                 pat_results = {
#                             args.study: {},
#                             args.treat: {}
#                             }

#                 acc = 0

#                 for i, (X, y) in enumerate(temp_loader):
#                     if test_locs[i][1] not in pat_results[test_locs[i][0]]:
#                         pat_results[test_locs[i][0]][test_locs[i][1]] = {'c': 0., 't': 0.}
#                     logits = model(X)
#                     acc = mean_accuracy(logits, y)
#                     pat_results[test_locs[i][0]][test_locs[i][1]]['t'] += test_weights[i]
#                     if acc == 1:
#                         pat_results[test_locs[i][0]][test_locs[i][1]]['c'] += test_weights[i]

#                 for typ in pat_results:
#                     for pat in pat_results[typ]:
#                         pat_results[typ][pat]['acc'] = pat_results[typ][pat]['c'] / pat_results[typ][pat]['t']

#                 for typ in pat_results:
#                     print (typ)
#                     for pat in pat_results[typ]:
#                         print (pat, pat_results[typ][pat])

#                 total_acc = 0
#                 total_count = 0
#                 pat_predictions = []
#                 pat_targets = []
#                 for typ in pat_results:
#                     typ_acc = 0
#                     typ_count = 0
#                     for pat in pat_results[typ]:
#                         true = typs[typ]
#                         other = 1 if true == 0 else 0
#                         pat_targets.append(true)
#                         pred = [0., 0.]
#                         if pat_results[typ][pat]['acc'] >= 0.5:
#                             typ_acc += 1
#                         pred[true] = pat_results[typ][pat]['acc']
#                         pred[other] = 1 - pat_results[typ][pat]['acc']
#                         pat_predictions.append(pred)
#                         typ_count += 1
#                     print (typ, typ_acc, typ_count, typ_acc / typ_count)
#                     file.write(f'cls: {typ}, correct: {typ_acc}, total: {typ_count}, acc: {typ_acc/typ_count}\n')
#                     total_acc += typ_acc
#                     total_count += typ_count
#                 print (total_acc, total_count, total_acc / total_count)

#                 predictions = torch.tensor(pat_predictions)
#                 targets = torch.tensor(pat_targets)

#                 acc_macro = MulticlassAccuracy(average="macro", num_classes=2)
#                 acc_macro.update(predictions, targets)
#                 acc_none = MulticlassAccuracy(average=None, num_classes=2)
#                 acc_none.update(predictions, targets)
#                 acc = MulticlassAccuracy(num_classes=2)
#                 acc.update(predictions, targets)
#                 auroc = MulticlassAUROC(num_classes=2)
#                 auroc.update(predictions, targets)
#                 auprc = MulticlassAUPRC(num_classes=2)
#                 auprc.update(predictions, targets)
#                 recall_macro = MulticlassRecall(num_classes=2, average='macro')
#                 recall_macro.update(predictions, targets)
#                 recall_none = MulticlassRecall(num_classes=2, average=None)
#                 recall_none.update(predictions, targets)
#                 recall_weighted = MulticlassRecall(num_classes=2, average='weighted')
#                 recall_weighted.update(predictions, targets)
#                 recall = MulticlassRecall(num_classes=2)
#                 recall.update(predictions, targets)
#                 precision_macro = MulticlassPrecision(num_classes=2, average='macro')
#                 precision_macro.update(predictions, targets)
#                 precision_none = MulticlassPrecision(num_classes=2, average=None)
#                 precision_none.update(predictions, targets)
#                 precision_weighted = MulticlassPrecision(num_classes=2, average='weighted')
#                 precision_weighted.update(predictions, targets)
#                 precision = MulticlassPrecision(num_classes=2)
#                 precision.update(predictions, targets)
#                 f1_macro = MulticlassF1Score(num_classes=2, average='macro')
#                 f1_macro.update(predictions, targets)
#                 f1_none = MulticlassF1Score(num_classes=2, average=None)
#                 f1_none.update(predictions, targets)
#                 f1_weighted = MulticlassF1Score(num_classes=2, average='weighted')
#                 f1_weighted.update(predictions, targets)
#                 f1 = MulticlassF1Score(num_classes=2)
#                 f1.update(predictions, targets)
#                 confusion = MulticlassConfusionMatrix(num_classes=2)
#                 confusion.update(predictions, targets)

#                 print (f'Mean test accuracy: {acc.compute().item()}')
#                 file.write(f'Mean test accuracy: {acc.compute().item()}\n')
#                 print (f'Class mean test accuracy: {acc_macro.compute().item()}')
#                 file.write(f'Class mean test accuracy: {acc_macro.compute().item()}\n')
#                 acc_none_vals = acc_none.compute()
#                 print (f'Individual class test accuracies: "0" - {acc_none_vals[0].item()}, "1" - {acc_none_vals[1].item()}')
#                 file.write(f'Individual class test accuracies: "0" - {acc_none_vals[0].item()}, "1" - {acc_none_vals[1].item()}\n')
#                 print (f'Test AUPRC: {auprc.compute().item()}')
#                 file.write(f'Test AUPRC: {auprc.compute().item()}\n')
#                 print (f'Test AUROC: {auroc.compute().item()}')
#                 file.write(f'Test AUROC: {auroc.compute().item()}\n')
#                 print (f'Mean test recall: {recall.compute().item()}')
#                 file.write(f'Mean test recall: {recall.compute().item()}\n')
#                 print (f'Class mean test recall: {recall_macro.compute().item()}')
#                 file.write(f'Class mean test recall: {recall_macro.compute().item()}\n')
#                 print (f'Class weighted mean test recall: {recall_weighted.compute().item()}')
#                 file.write(f'Class weighted mean test recall: {recall_weighted.compute().item()}\n')
#                 recall_none_vals = recall_none.compute()
#                 print (f'Individual class test recalls: "0" - {recall_none_vals[0].item()}, "1" - {recall_none_vals[1].item()}')
#                 file.write(f'Individual class test recalls: "0" - {recall_none_vals[0].item()}, "1" - {recall_none_vals[1].item()}\n')
#                 print (f'Mean test precision: {precision.compute().item()}')
#                 file.write(f'Mean test precision: {precision.compute().item()}\n')
#                 print (f'Class mean test precision: {precision_macro.compute().item()}')
#                 file.write(f'Class mean test precision: {precision_macro.compute().item()}\n')
#                 print (f'Class weighted mean test precision: {precision_weighted.compute().item()}')
#                 file.write(f'Class weighted mean test precision: {precision_weighted.compute().item()}\n')
#                 precision_none_vals = precision_none.compute()
#                 print (f'Individual class test precisions: "0" - {precision_none_vals[0].item()}, "1" - {precision_none_vals[1].item()}')
#                 file.write(f'Individual class test precisions: "0" - {precision_none_vals[0].item()}, "1" - {precision_none_vals[1].item()}\n')
#                 print (f'Mean test f1: {f1.compute().item()}')
#                 file.write(f'Mean test f1: {f1.compute().item()}\n')
#                 print (f'Class mean test f1: {f1_macro.compute().item()}')
#                 file.write(f'Class mean test f1: {f1_macro.compute().item()}\n')
#                 print (f'Class weighted mean test f1: {f1_weighted.compute().item()}')
#                 file.write(f'Class weighted mean test f1: {f1_weighted.compute().item()}\n')
#                 f1_none_vals = f1_none.compute()
#                 print (f'Individual class test f1s: "0" - {f1_none_vals[0].item()}, "1" - {f1_none_vals[1].item()}')
#                 file.write(f'Individual class test f1s: "0" - {f1_none_vals[0].item()}, "1" - {f1_none_vals[1].item()}\n')
#                 confuse_vals = confusion.compute()
#                 print (f'Confusion matrix ("true_pred"): "0_0" - {int(confuse_vals[0][0].item())}, "0_1" - {int(confuse_vals[0][1].item())}, "1_0" - {int(confuse_vals[1][0].item())}, "1_1" - {int(confuse_vals[1][1].item())}')
#                 file.write(f'Confusion matrix ("true_pred"): "0_0" - {int(confuse_vals[0][0].item())}, "0_1" - {int(confuse_vals[0][1].item())}, "1_0" - {int(confuse_vals[1][0].item())}, "1_1" - {int(confuse_vals[1][1].item())}\n')

#                 file.write(f'All patient results: {str(pat_results)}\n')
            file.close()
