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

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score


parser = argparse.ArgumentParser(description='Run EEG Spectral experiments')
parser.add_argument('--num_channels', type=int, default=19)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--remove_channels', type=bool, default=False)
parser.add_argument('--rm_ch', type=str, default='14_15')
parser.add_argument('--meta_dir', type=str,default='./meta_data')
parser.add_argument('--results_dir', type=str,default='./results_split')
parser.add_argument('--weights_dir', type=str,default='./weights_split')
parser.add_argument('--data_path', type=str,default='./data/mci_ctrlb_psd')
parser.add_argument('--sf', type=float, default=250.)
parser.add_argument('--contig_len', type=int, default=250)
parser.add_argument('--art_thresh', type=int, default=0)
parser.add_argument('--lowcut', type=float, default=4.)
parser.add_argument('--highcut', type=float, default=40.)
parser.add_argument('--min_contigs', type=int, default=3)
parser.add_argument('--include_evt', type=bool, default=False)
parser.add_argument('--balanced', type=bool, default=False)
parser.add_argument('--norm_type', type=str, default='minmax')
parser.add_argument('--study', type=str, default='mci')
parser.add_argument('--treat', type=str, default='ctrlb')
parser.add_argument('--model_type', type=str, default='mlp')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--model_conf_path', type=str, default='./grid_confs/conf')
parser.add_argument('--split', type=float, default=0.75)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--runs', type=int, default=30)
parser.add_argument('--study_run', type=int, default=22)

args = parser.parse_args(['--remove_channels', 'True'])


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


for contig_len in [250]:
    args.contig_len = contig_len
    
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

    print ('Channels:', channels)

    # Find best gmm n_components.

    overall_scores = np.zeros((11, 10))
    overall_sscores = np.zeros((11, 10))

    for r in range(10):

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

        args.num_psd = data[train_locs[0][0]][train_locs[0][1]][train_locs[0][2]].shape[1]

        n = get_norms(args, data, channels, typs)

        train_dataset = EEGSDataset(data_dict=data, locs=train_locs, study=args.study, treat=args.treat, norms=n)
    #         test_dataset = EEGSDataset(data_dict=data, locs=test_locs, study=args.study, treat=args.treat, norms=n)

        # Get Dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    #         test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # For GMM
        trainX, trainy, testX, testy = None, None, None, None

        for X, y in tqdm(train_dataset):
            if trainX is None:
                trainX, trainy = X.unsqueeze(0).cpu().numpy(), y.unsqueeze(0).cpu().numpy()
            else:
                trainX, trainy = np.concatenate((trainX, X.unsqueeze(0).cpu().numpy())), np.concatenate((trainy, y.unsqueeze(0).cpu().numpy()))

    #         for X, y in tqdm(test_dataset):
    #             if testX is None:
    #                 testX, testy = X.unsqueeze(0).cpu().numpy(), y.unsqueeze(0).cpu().numpy()
    #             else:
    #                 testX, testy = np.concatenate((testX, X.unsqueeze(0).cpu().numpy())), np.concatenate((testy, y.unsqueeze(0).cpu().numpy()))

        trainX = trainX.reshape(-1, trainX.shape[1] * trainX.shape[2])
    #         testX = testX.reshape(-1, testX.shape[1] * testX.shape[2])

        class_0_inds = np.where(trainy == 0)[0]
        trainX0 = trainX[class_0_inds]

        class_1_inds = np.where(trainy == 1)[0]
        trainX1 = trainX[class_1_inds]

        for comp in range(2, 13):

            
            gm0 = GaussianMixture(n_components=comp, random_state=0, verbose=1).fit(trainX0)
            gm1 = GaussianMixture(n_components=comp, random_state=0, verbose=1).fit(trainX1)
            
            labels0, labels1 = gm0.predict(trainX0), gm1.predict(trainX1)
            
            overall_sscores[comp - 4, r] = (silhouette_score(trainX0, labels0) + silhouette_score(trainX1, labels1)) / 2

            overall_scores [comp - 4, r] = (gm0.score(trainX0) + gm1.score(trainX1)) / 2
            print (contig_len, comp, r, overall_scores [comp - 4, r], overall_sscores[comp - 4, r])

    print (overall_scores, overall_sscores)
    np.save(f'{args.meta_dir}/gmmgs_scores_{args.study}_{args.treat}_{args.rm_ch_str}_{args.contig_len}.npy', overall_scores)
    np.save(f'{args.meta_dir}/gmmgs_sscores_{args.study}_{args.treat}_{args.rm_ch_str}_{args.contig_len}.npy', overall_sscores)