import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 

# transforms for data
data_transforms = [
    transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
]
data_transforms = transforms.Compose(data_transforms)

class MpiiLoader(Dataset):
    def __init__(self, data_dir='', transform=None):

        '''
        INPUTS
        data_dir: path to data
        '''
        self.data_dir = data_dir
        self.data = os.listdir(self.data_dir) # names for data_files
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pose_path = os.path.join(self.data_dir, self.data[idx]) # path to pose
        pose = np.load(pose_path)
        # transpose and flatten, reshape to 1D, stack [x, y]
        pose = pose.T.flatten() 

        if self.transform: # apply transforms to pose data
            pose = self.transform(pose)

        pose = pose[:, None, None] # add H and W dimensions -> [batch, channels, 1, 1]
        return pose

if __name__=='__main__':

    # # transforms for data
    # data_transforms = [
    #     transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    # ]

    # data_transforms = transforms.Compose(data_transforms)

    data_dir = 'data/data_2D/mpii_single/' # data directory
    dataset = MpiiLoader(data_dir=data_dir, transform=data_transforms) # instantiate class
    mpii_loader = DataLoader(dataset, batch_size=3, shuffle=True)


    for pose in mpii_loader:
        print("pose", pose)
        print("pose.shape", pose.shape)
        
        break


