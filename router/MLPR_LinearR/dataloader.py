from torch.utils import data
import numpy.linalg as la
import numpy as np
from scipy.io import loadmat
import random

class DatasetTrain(data.Dataset):
    
    'Characterizes a dataset for PyTorch'
    def __init__(self, datadir):

        data_x = np.load(datadir)['train_embed']
        self.data_x = data_x

        data_y = np.load(datadir)['train_score']
        self.data_y = data_y

        self.size = self.data_x.shape[0]

        print('******* train set size ******* {} {} '.format(data_x.shape, data_y.shape))
        
    def __len__(self):
        'Denotes the total number of samples'
        return self.size

    def __getitem__(self, index):

        return self.data_x[index], self.data_y[index]

class DatasetTest(data.Dataset):
    
    'Characterizes a dataset for PyTorch'
    def __init__(self, datadir):
        
        data_x = np.load(datadir)['test_embed']
        self.data_x = data_x

        data_y = np.load(datadir)['test_score']
        self.data_y = data_y

        self.size = self.data_x.shape[0]

        print('******* test set size ******* {} {} '.format(data_x.shape, data_y.shape))
        
    def __len__(self):
        'Denotes the total number of samples'
        return self.size
    
    def __getitem__(self, index):
        
        return self.data_x[index], self.data_y[index]

if __name__ == '__main__':
    print()
