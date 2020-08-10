from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import PIL.Image as Image


class RSNADataset(Dataset):

    def __init__(self, root_dir, transform=None, train=True):
        '''
        Args:
            root_dir (str): the root for the data
            transform (callable, None): transformation for the data
            train (bool, True): if set returns the training dataset
            '''

        self.dirname = 'boneage-training-dataset' if train else 'boneage-test-dataset'
        csv_fname = os.path.join(root_dir, f'{self.dirname}.csv')
        self.targets = pd.read_csv(csv_fname)
        self.root_path = os.path.join(root_dir, self.dirname, self.dirname)
        self.transform = transform

        return

    def __len__(self):

        return len(self.targets)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_path, f'{self.targets.iloc[idx, 0]}.png')

        image = Image.open(img_name)

        age = self.targets.iloc[idx, 1]
        male = self.targets.iloc[idx, 2]

        if self.transform is not None:
            image = self.transform(image)

        return image, male, age





