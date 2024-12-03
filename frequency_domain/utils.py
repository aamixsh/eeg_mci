import os
import scipy.signal as signal
import torch.nn as nn
import torch
import pickle
from tqdm import tqdm
import numpy as np
import json
import ast

# Reads model configurations from config file
def read_grid_confs(path):
    file = open(path)
    confs = []
    for line in file:
        name, conf = line.strip().split(':')
        conf = [int(x) for x in conf.strip().split(',')]
        confs.append((name, conf))
    return confs


# Returns locs from data for usage later.
def get_clean_locs(args, all_data, typs, new=False):
    if not new and os.path.exists(f'{args.meta_dir}/locs_{args.study}_{args.treat}_{args.rm_ch_str}_{args.contig_len}.pkl'):
        locs = pickle.load(open(f'{args.meta_dir}/locs_{args.study}_{args.treat}_{args.rm_ch_str}_{args.contig_len}.pkl', 'rb'))
    else:
        locs = []

        for typ in typs:
            for pid, datum in tqdm(enumerate(all_data[typ])):
                for cid, _ in enumerate(datum):
                    locs.append([typ, pid, cid])
        if not new:
            pickle.dump(locs, open(f'{args.meta_dir}/locs_{args.study}_{args.treat}_{args.rm_ch_str}_{args.contig_len}.pkl', 'wb'))
    return locs


# Get class-wise data info.
def get_info(locs, args):
    info = {args.study: [], args.treat: []}
    prev = args.study
    patient = 0
    this_count = 0

    for loc in locs:
        if loc[0] == prev and loc[1] == patient:
            this_count += 1
        else:
            if this_count >= args.min_contigs:
                info[prev].append((patient, this_count))
            if prev != loc[0]:
                prev = loc[0]
                patient = 0
            else:
                patient += 1
            while (patient != loc[1]):
                patient += 1
            this_count = 1
    if this_count >= args.min_contigs:
        info[prev].append((patient, this_count))
        
    return info


# Split locs into train and test.
def get_train_test_locs(args, data, new_locs):
    
    patients_inds = {}
    train_locs = []
    test_locs = []
    for typ in data:
        if typ not in patients_inds:
            patients_inds[typ] = []
        for pid in range(len(data[typ])):
            if len(data[typ][pid]):
                patients_inds[typ].append(pid)
        patients_inds[typ] = np.array(patients_inds[typ])
        np.random.shuffle(patients_inds[typ])

    min_typ = args.study if len(patients_inds[args.study]) < len(patients_inds[args.treat]) else args.treat
    max_typ = args.study if min_typ == args.treat else args.treat
    len_max = int(len(patients_inds[max_typ]) * args.split)
    len_min = int(len(patients_inds[min_typ]) * args.split)
    if args.balanced:
        subsampled_pats = np.random.choice(patients_inds[max_typ][:len_max], len_min, replace=False)

    train_pats = {max_typ: set(), min_typ: set()}
    test_pats = {max_typ: set(), min_typ: set()}
    if args.balanced:
        train_pats[max_typ] = set(subsampled_pats)
        test_pats[max_typ] = set(patients_inds[max_typ][len_max:])
        for pat in patients_inds[max_typ][:len_max]:
            if pat not in train_pats[max_typ]:
                test_pats[max_typ].add(pat)
    else:
        train_pats[max_typ] = set(patients_inds[max_typ][:len_max])
        test_pats[max_typ] = set(patients_inds[max_typ][len_max:])
        
    train_pats[min_typ] = set(patients_inds[min_typ][:len_min])
    test_pats[min_typ] = set(patients_inds[min_typ][len_min:])
    
    for loc in new_locs:
        if loc[0] == max_typ and loc[1] in train_pats[max_typ]:
            train_locs.append(loc)
        elif loc[0] == max_typ and loc[1] in test_pats[max_typ]:
            test_locs.append(loc)
        elif loc[0] == min_typ and loc[1] in train_pats[min_typ]:
            train_locs.append(loc)
        elif loc[0] == min_typ and loc[1] in test_pats[min_typ]:
            test_locs.append(loc)
        
    return train_locs, test_locs


def get_norms(args, data, channels, typs):
    # For normalization
    if args.norm_type == 'meanstd':
        channel_means = np.zeros((len(channels), args.num_psd))
        channel_stds = np.zeros((len(channels), args.num_psd))

        count = 0

        for typ in data:
            for pid in tqdm(range(len(data[typ]))):
                if not len(data[typ][pid]):
                    continue
                    
                channel_means += np.mean(data[typ][pid], 0)
                channel_stds += np.std(data[typ][pid], 0)
            count += len(data[typ])
        channel_means /= count
        channel_stds /= count

        norms = {'mean': torch.tensor(channel_means).type(torch.float32), 'std': torch.tensor(channel_stds).type(torch.float32)}

    elif args.norm_type == 'minmax':

        channel_maxs = np.full((len(channels), args.num_psd), -np.inf)
        channel_mins = np.full((len(channels), args.num_psd), np.inf)

        count = 0

        for typ in typs:
            for pid in tqdm(range(len(data[typ]))):
                if not len(data[typ][pid]):
                    continue
                channel_mins = np.minimum(np.min(data[typ][pid], 0), channel_mins)
                channel_maxs = np.maximum(np.max(data[typ][pid], 0), channel_maxs)
                count += 1

        norms = {'mins': torch.tensor(channel_mins).type(torch.float32), 'maxs': torch.tensor(channel_maxs).type(torch.float32)}
    
    else:
        norms = None
    return norms


# Bandpass filters
def butter_bandpass(lowcut, highcut, fs, order=5):
    return signal.butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

# Loss/accuracy functions.
def nll(logits, y, reduction='mean', weight=None):
    return nn.functional.cross_entropy(logits, y, reduction=reduction, weight=weight)

def mse(logits, y, reduction='mean', weight=None):
    return nn.functional.mse_loss(logits, y, reduction=reduction, weight=weight)

def mean_accuracy(logits, y, reduce='sum'):
    preds = torch.argmax(logits, dim=1)
    if reduce == 'mean':
        return torch.count_nonzero(preds == y) / len(preds)
    else:
        return torch.count_nonzero(preds == y)

# For debugging previous runs.
def get_train_test_locs_from_pats(args, data, new_locs, pats):
    idset = set()
    for split in pats:
        for typ in pats[split]:
            for p in pats[split][typ]:
                idset.add(split + '_' + typ + '_' + str(p))
    
    train_locs = []
    test_locs = []
    for loc in new_locs:
        if 'train_' + loc[0] + '_' + str(loc[1]) in idset:
            train_locs.append(loc)
        else:
            test_locs.append(loc)
    return train_locs, test_locs   

def get_train_test_pats_from_run(path, run):
    file = open(path)
    train_pats = []
    test_pats = []
    for line in file:
        if line.startswith('Train patients: '):
#             print (line[16:].strip().replace("'", "\""))
            train_pats.append(ast.literal_eval(line[16:].strip()))
        if line.startswith('Test patients: '):
            test_pats.append(ast.literal_eval(line[15:].strip()))
            
    return train_pats[run], test_pats[run]