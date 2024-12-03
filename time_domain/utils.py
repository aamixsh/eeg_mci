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
        conv_conf, fc_conf = conf.split('<>')
        conv_confs = []
        for c in conv_conf.split(';'):
            conv_confs.append([int(x) for x in c.strip().split(',')])
        fc_confs = [int(x) for x in fc_conf.strip().split(',')]
        conf = [conv_confs, fc_confs]
        confs.append((name, conf))
    return confs


# Returns locs from data for usage later.
def get_clean_locs3(args, all_data, typs, new=False):
    if not new and os.path.exists(f'{args.meta_dir}/locs_{args.study1}_{args.study2}_{args.treat}_{args.rm_ch_str}_{args.contig_len}.pkl'):
        locs = pickle.load(open(f'{args.meta_dir}/locs_{args.study1}_{args.study2}_{args.treat}_{args.rm_ch_str}_{args.contig_len}.pkl', 'rb'))
    else:
        locs = []

        for typ in typs:
            for pid, datum in tqdm(enumerate(all_data[typ])):
                for cid, _ in enumerate(datum):
                    locs.append([typ, pid, cid])
        if not new:
            pickle.dump(locs, open(f'{args.meta_dir}/locs_{args.study1}_{args.study2}_{args.treat}_{args.rm_ch_str}_{args.contig_len}.pkl', 'wb'))
    return locs

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
def get_info3(locs, args):
    info = {args.study1: [], args.study2: [], args.treat: []}
    prev = args.study1
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
def get_train_test_locs3(args, data, new_locs):
    
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

    min_typ = args.study1
    max_typ = args.study2
    if len(patients_inds[args.study2]) < len(patients_inds[min_typ]):
        min_type = args.study2
        max_typ = args.study1
    if len(patients_inds[args.treat]) < len(patients_inds[min_typ]):
        min_type = args.treat
    else:
        if len(patients_inds[args.treat]) > len(patients_inds[max_typ]):
            max_typ = args.treat
            
    mid_typ = args.treat
    if min_typ == mid_typ or max_typ == mid_typ:
        mid_type = args.study1
        if min_typ == mid_typ or max_typ == mid_typ:
            mid_typ = args.study2

    len_max = int(len(patients_inds[max_typ]) * args.split)
    len_mid = int(len(patients_inds[mid_typ]) * args.split)
    len_min = int(len(patients_inds[min_typ]) * args.split)
    
    if args.balanced:
        subsampled_pats_max = np.random.choice(patients_inds[max_typ][:len_max], len_min, replace=False)
        subsampled_pats_mid = np.random.choice(patients_inds[mid_typ][:len_mid], len_min, replace=False)

    train_pats = {max_typ: set(), min_typ: set(), mid_typ: set()}
    test_pats = {max_typ: set(), min_typ: set(), mid_typ: set()}
    if args.balanced:
        train_pats[max_typ] = set(subsampled_pats_max)
        train_pats[mid_typ] = set(subsampled_pats_mid)
        test_pats[max_typ] = set(patients_inds[max_typ][len_max:])
        test_pats[mid_typ] = set(patients_inds[mid_typ][len_mid:])
        for pat in patients_inds[max_typ][:len_max]:
            if pat not in train_pats[max_typ]:
                test_pats[max_typ].add(pat)
                
        for pat in patients_inds[mid_typ][:len_mid]:
            if pat not in train_pats[mid_typ]:
                test_pats[mid_typ].add(pat)
    else:
        train_pats[max_typ] = set(patients_inds[max_typ][:len_max])
        test_pats[max_typ] = set(patients_inds[max_typ][len_max:])
        train_pats[mid_typ] = set(patients_inds[mid_typ][:len_mid])
        test_pats[mid_typ] = set(patients_inds[mid_typ][len_mid:])
        
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
        elif loc[0] == mid_typ and loc[1] in train_pats[mid_typ]:
            train_locs.append(loc)
        elif loc[0] == mid_typ and loc[1] in test_pats[mid_typ]:
            test_locs.append(loc)
        
    return train_locs, test_locs


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



def get_norms(args, data, channels, typs, return_tensors=True):
    # For normalization
    if args.norm_type == 'meanstd':
        channel_means = np.zeros(len(channels))
        channel_stds = np.zeros(len(channels))

        count = 0

        for typ in data:
            for pid in tqdm(range(len(data[typ]))):
                if not len(data[typ][pid]):
                    continue
                    
                channel_means += np.mean(data[typ][pid], (0, 2))
                channel_stds += np.std(data[typ][pid], (0, 2))
            count += len(data[typ])
        
        channel_means /= count
        channel_stds /= count

        if return_tensors:
            norms = {'mean': torch.tensor(channel_means).type(torch.float32), 'std': torch.tensor(channel_stds).type(torch.float32)}
        else:
            norms = {'mean': channel_means, 'std': channel_stds}

    elif args.norm_type == 'minmax':

        channel_maxs = np.full((len(channels)), -np.inf)
        channel_mins = np.full((len(channels)), np.inf)

        for typ in typs:
            for pid in tqdm(range(len(data[typ]))):
                if not len(data[typ][pid]):
                    continue
                channel_mins = np.minimum(np.min(data[typ][pid], (0, 2)), channel_mins)
                channel_maxs = np.maximum(np.max(data[typ][pid], (0, 2)), channel_maxs)

        if return_tensors:
            norms = {'mins': torch.tensor(channel_mins).type(torch.float32), 'maxs': torch.tensor(channel_maxs).type(torch.float32)}
        else:
            norms = {'mins': channel_mins, 'maxs': channel_maxs}
    
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
        elif 'test_' + loc[0] + '_' + str(loc[1]) in idset:
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

def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).to(mat_a.device)
    
    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)
    
    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)