from torch.utils import data
import numpy as np

class cubeDataset(data.Dataset):
    def __init__(self, train=True):
        self.cube = np.load('data/shuffled/32/img_segmented.npy')   # [123, 64, 32, 32, 32]
        self.label = np.load('data/shuffled/32/labels.npy')
        # self.node_mask = np.load('shuffled/node_mask.npy')  # [123, 64, 32, 32, 32]
        if train:
            self.cube = self.cube[:86]
            self.label = self.label[:86]
        else:
            self.cube = self.cube[86:]
            self.label = self.label[86:]
        
    def __getitem__(self, index):
        return self.cube[index], self.label[index]

    def __len__(self):
        return self.cube.shape[0]
            

        