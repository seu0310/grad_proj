import os
import utils.config as config
import numpy as np
from torch.utils.data import Dataset

    
class CelebaDataset(Dataset):
    def __init__(self, split=0):
        name_dic = {0: 'train', 1: 'val', 2: 'test'}
        src_dir = config.celeba_path
        if split == 1:
            src_dir = config.celeba_val_path


        self.features = np.load(os.path.join(src_dir, f'{name_dic[split]}_feats.npy'))
        self.bias = np.load(os.path.join(src_dir, f'{name_dic[split]}_bias.npy'))
        self.targets = np.load(os.path.join(src_dir, f'{name_dic[split]}_targets.npy'))


        self.class_sample_count = np.array(
            [len(np.where(self.targets == t)[0]) for t in np.unique(self.targets)]
        )

        unique_bias = np.unique(self.bias)
        bias_to_idx = {b: i for i, b in enumerate(unique_bias)}
        bias_indices = np.array([bias_to_idx[b] for b in self.bias])
        self.group_ids = (self.targets * len(unique_bias) + bias_indices).astype(np.int64)

    def __getitem__(self, index):
        img = self.features[index]
        label = self.targets[index]
        bias = self.bias[index]
        return index, img, label, bias, index

    def __len__(self):
        return self.features.shape[0]


class WaterBirds(Dataset):
    def __init__(self, split):

        src_dir = config.waterbirds_path
        if split == 'val' or split == 'test':
            src_dir = config.waterbirds_val_path
        
        self.features = np.load(os.path.join(src_dir, f"{split}_feats.npy")).astype(np.float32)
        self.targets = np.load(os.path.join(src_dir, f"{split}_targets.npy"))
        self.bias = np.load(os.path.join(src_dir, f"{split}_bias.npy"))

        self.group_ids = self.targets * 2 + self.bias

        if split == 'train':
            self.class_sample_count = np.array(
                [len(np.where(self.targets == t)[0]) for t in np.unique(self.targets)]
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):

        return index, self.features[index], self.targets[index], self.bias[index], index

    def get_targets(self):
        return self.targets

    def get_biases(self):
        return self.bias

    def get_group_ids(self):
        return self.group_ids

