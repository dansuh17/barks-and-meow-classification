import numpy as np
import pandas as pd
import os
import torch
from torch import nn, optim
from torch.utils import data
from scipy.io import wavfile

INPUT_ROOT = '../input'
TRAIN_TEST_SPLIT_CSV = os.path.join(INPUT_ROOT, 'train_test_split.csv')
AUDIO_DIR = os.path.join(INPUT_ROOT, 'cats_dogs')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(os.listdir("../input"))


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


class CatsDogsModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, bias=False),
            nn.InstanceNorm1d(8),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=5, stride=3, padding=1, groups=8, bias=False),
            nn.InstanceNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=5, stride=3, dilation=2, groups=8, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, dilation=2, groups=8, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, dilation=2, groups=16, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, dilation=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, stride=2, dilation=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=108, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

        self.apply(self.init_weights)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 108)
        return self.linear(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)


class CatsDogsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(8),
            nn.Dropout(0.5),

            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(8),
            nn.Dropout(0.5),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5),

            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, stride=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=333, out_features=100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(100, 1),
        )

        self.apply(self.init_weights)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 333)
        return self.linear(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)


class CatsDogsModel3(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.InstanceNorm1d(32),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.InstanceNorm1d(64),
            nn.Dropout(0.5),
        )

        # reducing dimension on the time axis
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=(1, 2), padding=(1, 2), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
        )

        # accumulating features
        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout(0.5),
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=512, out_features=100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(100, 1),
        )

        self.apply(self.init_weights)

    def forward(self, x):
        x = self.filter(x)
        x = x.view(-1, 1, 64, 4000)
        x = self.net(x)
        x = self.net2(x)
        x = x.view(-1, 512)
        return self.linear(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)


class ResBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.InstanceNorm1d(in_channels),
            nn.Dropout(0.5),

            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm1d(out_channels),
        )

        self.activation = nn.ELU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.conv1(x))


class ResBlock1dDownSamp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(in_channels),
            nn.Dropout(0.5),

            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        self.downsampler = nn.Sequential(
            # TODO: dilated?
            nn.AvgPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
        )

        self.activation = nn.ELU(inplace=True)

    def forward(self, x):
        return self.activation(self.downsampler(x) + self.conv1(x))


class CatsDogsRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Sequential(
            ResBlock1d(1, 4),
            ResBlock1dDownSamp(4, 8),
            ResBlock1d(8, 8),
            ResBlock1dDownSamp(8, 8),
            nn.Dropout(0.7),
            ResBlock1d(8, 8),
            ResBlock1dDownSamp(8, 8),
            ResBlock1d(8, 8),
            ResBlock1dDownSamp(8, 16),
            nn.Dropout(0.7),
            ResBlock1d(16, 16),
            ResBlock1dDownSamp(16, 16),
            ResBlock1d(16, 16),
            ResBlock1dDownSamp(16, 32),
            nn.Dropout(0.7),
            ResBlock1d(32, 32),
            ResBlock1dDownSamp(32, 32),
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=4000, out_features=1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=1000, out_features=1),
        )

        self.apply(self.init_weights)

    def forward(self, x):
        x = self.filter(x)
        x = x.view(-1, 4000)
        return self.linear(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)


class CatsDogsResSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Sequential(
            ResBlock1d(1, 8),
            ResBlock1dDownSamp(8, 8),
            ResBlock1dDownSamp(8, 8),
            ResBlock1dDownSamp(8, 16),
            ResBlock1dDownSamp(16, 16),
            ResBlock1dDownSamp(16, 32),
            ResBlock1dDownSamp(32, 32),
            ResBlock1dDownSamp(32, 32),
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=4000, out_features=1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=1000, out_features=1),
        )

        self.apply(self.init_weights)

    def forward(self, x):
        x = self.filter(x)
        x = x.view(-1, 4000)
        return self.linear(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)

# TODO: densenet, squeeze-and-excite, shift-invariant networks


if __name__ == '__main__':
    lr = 1e-4
    epoch = 200

    model = nn.DataParallel(CatsDogsRes()).to(DEVICE)

    train_dataset = CatsDogsDataset(AUDIO_DIR, TRAIN_TEST_SPLIT_CSV, True)
    test_dataset = CatsDogsDataset(AUDIO_DIR, TRAIN_TEST_SPLIT_CSV, False)

    train_dataloader = data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=64, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    loss_func = nn.BCEWithLogitsLoss()

    for i in range(epoch):
        for targets, samples in train_dataloader:
            targets = targets.type(torch.FloatTensor).to(DEVICE)
            samples = samples.to(DEVICE)
            out = torch.squeeze(model(samples))
            loss = loss_func(out, targets)

            out_bool = torch.sigmoid(out) > 0.5
            target_bool = targets > 0.5
            pred = (out_bool == target_bool).type(torch.FloatTensor)
            accuracy = pred.sum() / pred.numel()
            print(f'epoch: {i}, loss: {loss}, accuracy: {accuracy}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_accs = []
        with torch.no_grad():
            for targets, samples in test_dataloader:
                targets = targets.type(torch.FloatTensor).to(DEVICE)
                samples = samples.to(DEVICE)
                out = torch.squeeze(model(samples))
                loss = loss_func(out, targets)

                out_bool = torch.sigmoid(out) > 0.5
                target_bool = targets > 0.5
                pred = (out_bool == target_bool).type(torch.FloatTensor)
                accuracy = pred.sum() / pred.numel()

                test_accs.append(accuracy)
                print(f'TEST: epoch: {i}, loss: {loss}, accuracy: {accuracy}')

                # save any that are mispredicted
                if i == epoch - 1:
                    for target_class, samp, pred_score in zip(targets, samples, out):
                        pred_class = 1 if pred_score > 0.5 else 0
                        if target_class != pred_class:
                            as_ = 'cat' if pred_class == CatsDogsDataset.LABEL_CAT else 'dog'
                            wavfile.write(f'mispred_{as_}_{pred_score:04f}.wav', 16000, samp.cpu().detach().numpy().T)
                            print(f'misprediction ({as_}): {pred_score:04f}')

        print(f'Test Accuracy: {sum(test_accs) / len(test_accs)}')

    # save module
    print('Saving model')
    torch.save(model.module.state_dict(), 'model.pt')
