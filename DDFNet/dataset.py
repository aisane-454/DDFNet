import os
import numpy as np
from torch.utils.data import Dataset

class HypertenseDataset(Dataset):
    def __init__(self, base_dir="./datasets/data_218_hypertension_longer", split="train", transform=None, channel_idx=0):
        self._base_dir = base_dir
        self.transform = transform
        self.label = np.load(os.path.join(base_dir, "label.npy"))
        self.image = np.load(os.path.join(base_dir, "data.npy"))
        self.channel_idx = channel_idx

        split_path = os.path.join(self._base_dir, f"{split}.list")
        with open(split_path) as file:
            lines = [int(item) for item in file.readlines()]
            self.label = self.label[lines]
            self.image = self.image[lines, :, :]
        print(f"total {self.label.size} samples")
    
    def __len__(self):
        return self.label.size
    
    def do_fft(self, all_channel_data):
        """
        Do fft in each channel for all channels.
        Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
        Output: FFT result with dimension N x M. N denotes number of channel and M denotes number of FFT data from each channel.
        """
        data_fft = map(lambda x: np.fft.fft(x), all_channel_data)
        return np.array(list(data_fft))
    
    def __getitem__(self, idx):
        image = self.image[idx]
        label = self.label[idx]
        # image = self.do_fft(image)[self.channel_idx]
        # image = np.expand_dims(image, axis=0)

        if self.transform:
            image = self.transform(image)
        return image, label
