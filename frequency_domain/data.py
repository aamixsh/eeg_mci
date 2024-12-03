import torch
from torch.utils.data import Dataset

class EEGSDataset(Dataset):

    """
    EEG Spectral Dataset
    """

    def __init__(self,
                 data_dict,
                 locs,
                 study,
                 treat,
                 norms=None,
                 cuda=True,
                 mask=None):
        """
        Args:
            data_dict (dict): filtered data dict
            locs (list): Locations of clean targets
            evt (bool): include EVT signal
            norms (dict): dict of normalization parameters
        """
        self.data = data_dict
        self.typ = {study: 0, treat: 1}
        self.locs = locs
        self.norms = norms
        if self.norms:
            if 'mean' in self.norms:
                self.normalize = self.normalize_mean
            else:
                self.normalize = self.normalize_min_max
        else:
            self.normalize = lambda x: x
            
        self.cuda = cuda
        self.mask = mask

    def normalize_mean(self, X):
        for i in range(X.shape[0]):
            X[i] = (X[i] - self.norms['mean'][i]) / self.norms['std'][i]
        return X
            
    def normalize_min_max(self, X):
        return (X - self.norms['mins']) / (self.norms['maxs'] - self.norms['mins'])
        
        
    def __len__(self):
        """

        """
        return len(self.locs)

    def __getitem__(self,
                    idx):
        """

        """
        X = self.normalize(torch.tensor(self.data[self.locs[idx][0]][self.locs[idx][1]][self.locs[idx][2]]).type(torch.float32))
        y = torch.tensor(self.typ[self.locs[idx][0]]).type(torch.LongTensor)
            
        if self.mask is not None:
            X = X * self.mask
        if self.cuda:
            return X.cuda(), y.cuda()
        return X, y
