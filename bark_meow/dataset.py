import os
from collections import namedtuple
from torch.utils import data
import numpy as np
from scipy.io import wavfile


# tuple representing the label and the full path of file containing audio segment
AudioData = namedtuple('AudioFilePath', ['label', 'path'])


class BarkMeowDB(data.Dataset):
    """
    Dataset representing a set of audio chunks,
    each containing either dog barking or cat meowing sounds.
    """
    def __init__(self, audio_root: str):
        super().__init__()
        self.audio_root = audio_root
        self.class_label_name = {}
        # contains list of (class_label, audio_path)
        self.audio_data_list = []

        # label is the number and the name is its string representation
        for class_label, class_name in enumerate(os.listdir(self.audio_root)):
            self.class_label_name[class_label] = class_name

            class_audio_path = os.path.join(self.audio_root, class_name)
            for fname in os.listdir(class_audio_path):
                audio_path = os.path.join(class_audio_path, fname)

                self.audio_data_list.append(
                    AudioData(label=class_label, path=audio_path))

    def __getitem__(self, idx: int):
        # read the audio data
        audio_data: AudioData = self.audio_data_list[idx]
        label, audio_path = audio_data.label, audio_data.path
        _, data = wavfile.read(audio_path)

        # add dimension at the front to make shape of: (1, 16000)
        return label, data[np.newaxis, :]

    def __len__(self):
        return len(self.audio_data_list)


if __name__ == '__main__':
    ds = BarkMeowDB('./barkmeow_db')
