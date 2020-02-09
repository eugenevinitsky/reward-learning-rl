import os

from skimage import io
import torch
from torch.utils.data import Dataset

class BottleDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, success_dir, fail_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.success_dir = success_dir
        self.fail_dir = fail_dir

        file_names = [file for file in os.listdir(self.success_dir) if 'jpg' in file]
        # if the idx is less than this, we look in the success directory
        self.file_cutoff = len(file_names)

        file_names.extend( [file for file in os.listdir(self.fail_dir) if 'jpg' in file ])
        self.file_names = file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx < self.file_cutoff:
            try:
                img_path = os.path.join(self.success_dir, self.file_names[idx])
            except:
                import ipdb; ipdb.set_trace()
            label = 1
        else:
            img_path = os.path.join(self.fail_dir, self.file_names[idx])
            label = 0

        image = io.imread(img_path)
        sample = (image, label)

        return sample
