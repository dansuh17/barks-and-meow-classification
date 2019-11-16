import os
from torch.utils import data
import numpy as np
from scipy.io import wavfile


class BarkMeowDB(data.Dataset):
    def __init__(self, audio_root: str):
        super().__init__()
        self.audio_root = audio_root
        self.class_label_name = {}
        # contains list of (class_label, audio_path)
        self.data_paths = []

        for class_label, class_name in enumerate(os.listdir(self.audio_root)):
            self.class_label_name[class_label] = class_name

            class_audio_path = os.path.join(self.audio_root, class_name)
            for fname in os.listdir(class_audio_path):
                audio_path = os.path.join(class_audio_path, fname)

                self.data_paths.append((class_label, audio_path))

    def __getitem__(self, idx):
        # read the audio data
        label, audio_path = self.data_paths[idx]
        _, audio_data = wavfile.read(audio_path)

        # add dimension at the front to make shape of: (1, 16000)
        return label, audio_data[np.newaxis, :]

    def __len__(self):
        return len(self.data_paths)


if __name__ == '__main__':
    ds = BarkMeowDB('./barkmeow_db')
