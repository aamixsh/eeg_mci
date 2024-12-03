import torch
from torch.utils.data import Dataset
# import tensorflow as tf  # Add TensorFlow import

class EEGDataset(Dataset):
    """
    EEGDataset
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
        for i in range(X.shape[0]):
            X[i] = (X[i] - self.norms['mins'][i]) / (self.norms['maxs'][i] - self.norms['mins'][i])
        return X
        
        
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
    
    
class EEGDataset3(Dataset):
    """
    EEGDataset
    """

    def __init__(self,
                 data_dict,
                 locs,
                 study1,
                 study2,
                 treat,
                 norms=None,
                 cuda=True,
                 mask=None):
        """
        Args:
            data_dict (dict): filtered data dict
            locs (list): Locations of clean targets
            norms (dict): dict of normalization parameters
        """
        self.data = data_dict
        self.typ = {study1: 0, study2: 1, treat: 2}
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
        for i in range(X.shape[0]):
            X[i] = (X[i] - self.norms['mins'][i]) / (self.norms['maxs'][i] - self.norms['mins'][i])
        return X
        
        
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
    


# class EEGDatasetTF(tf.data.Dataset):
#     """
#     EEGDataset for TensorFlow
#     """

#     def __init__(self,
#                  data_dict,
#                  locs,
#                  study,
#                  treat,
#                  norms=None,
#                  mask=None):
#         self.data = data_dict
#         self.typ = {study: 0, treat: 1}
#         self.locs = locs
#         self.norms = norms
#         if self.norms:
#             if 'mean' in self.norms:
#                 self.normalize = self.normalize_mean
#             else:
#                 self.normalize = self.normalize_min_max
#         else:
#             self.normalize = lambda x: x

#         self.mask = mask

#     def normalize_mean(self, X):
#         for i in range(X.shape[0]):
#             X[i] = (X[i] - self.norms['mean'][i]) / self.norms['std'][i]
#         return X

#     def normalize_min_max(self, X):
#         for i in range(X.shape[0]):
#             X[i] = (X[i] - self.norms['mins'][i]) / (self.norms['maxs'][i] - self.norms['mins'][i])
#         return X

#     def __len__(self):
#         return len(self.locs)

#     def __getitem__(self, idx):
#         X = self.normalize(tf.convert_to_tensor(self.data[self.locs[idx][0]][self.locs[idx][1]][self.locs[idx][2]], dtype=tf.float32))
#         y = tf.convert_to_tensor(self.typ[self.locs[idx][0]], dtype=tf.int64)
        
#         if self.mask is not None:
#             X = X * self.mask
#         return X, y


# class EEGDataset3TF(tf.data.Dataset):
#     """
#     EEGDataset3 for TensorFlow
#     """

#     def __init__(self,
#                  data_dict,
#                  locs,
#                  study1,
#                  study2,
#                  treat,
#                  norms=None,
#                  mask=None):
#         self.data = data_dict
#         self.typ = {study1: 0, study2: 1, treat: 2}
#         self.locs = locs
#         self.norms = norms
#         if self.norms:
#             if 'mean' in self.norms:
#                 self.normalize = self.normalize_mean
#             else:
#                 self.normalize = self.normalize_min_max
#         else:
#             self.normalize = lambda x: x

#         self.mask = mask

#     def normalize_mean(self, X):
#         for i in range(X.shape[0]):
#             X[i] = (X[i] - self.norms['mean'][i]) / self.norms['std'][i]
#         return X

#     def normalize_min_max(self, X):
#         for i in range(X.shape[0]):
#             X[i] = (X[i] - self.norms['mins'][i]) / (self.norms['maxs'][i] - self.norms['mins'][i])
#         return X

#     def __len__(self):
#         return len(self.locs)

#     def __getitem__(self, idx):
#         X = self.normalize(tf.convert_to_tensor(self.data[self.locs[idx][0]][self.locs[idx][1]][self.locs[idx][2]], dtype=tf.float32))
#         y = tf.convert_to_tensor(self.typ[self.locs[idx][0]], dtype=tf.int64)
        
#         if self.mask is not None:
#             X = X * self.mask
#         return X, y