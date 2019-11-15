import os
from torch.utils import data
import numpy as np
import pandas as pd
from scipy.io import wavfile


class CatsDogsDataset(data.Dataset):
    LABEL_CAT = 0
    LABEL_DOG = 1

    def __init__(self, audio_root: str, csv_file: str, is_train: bool):
        super().__init__()
        self.audio_root = audio_root

        prefix = 'train' if is_train else 'test'

        df = pd.read_csv(csv_file)
        train_cats = df[prefix + '_cat'].dropna()  # drop NaNs
        train_dogs = df[prefix + '_dog'].dropna()  # drop NaNs

        int16_max = np.iinfo(np.int16).max

        data = []
        for fname in train_cats:
            sr, audio_data = wavfile.read(os.path.join(self.audio_root, fname))
            # modify range to [-1, 1]
            audio_data = audio_data.astype(np.float16) / int16_max
            length = audio_data.shape[0]
            for i in range(0, length, 4000):
                if i + 16000 >= length:
                    break

                segment = audio_data[i:i + 16000][np.newaxis, :]

                # ignore silent data
                rms = np.sqrt(np.mean(segment**2))
                if rms < 0.02:  # TODO: adjust threshold
                    # wavfile.write(f'rms{rms}' + fname, sr, segment.T)
                    continue

                # normalize to zero mean
                segment -= np.mean(segment)
                # max normalize
                max_abs = max(np.abs(segment.max()), np.abs(segment.min()))
                segment /= max_abs

                data.append((self.LABEL_CAT, segment))

        for fname in train_dogs:
            sr, audio_data = wavfile.read(os.path.join(self.audio_root, fname))
            audio_data = audio_data.astype(np.float16) / int16_max
            length = audio_data.shape[0]
            for i in range(0, length, 4000):
                if i + 16000 >= length:
                    break

                segment = audio_data[i:i + 16000][np.newaxis, :]

                # ignore silent data
                rms = np.sqrt(np.mean(segment**2))

                if rms < 0.02:  # TODO: adjust threshold
                    # wavfile.write(f'rms{rms}' + fname, sr, segment.T)
                    continue

                # normalize to zero mean
                segment -= np.mean(segment)
                # max normalize
                max_abs = max(np.abs(segment.max()), np.abs(segment.min()))
                segment /= max_abs
                data.append((self.LABEL_DOG, segment))
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
