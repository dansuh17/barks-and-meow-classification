import os
import torch
from torch import nn, optim
from torch.utils import data
from scipy.io import wavfile
from .model import CatsDogsDenseNet
from .dataset import CatsDogsDataset

INPUT_ROOT = '../input'
TRAIN_TEST_SPLIT_CSV = os.path.join(INPUT_ROOT, 'train_test_split.csv')
AUDIO_DIR = os.path.join(INPUT_ROOT, 'cats_dogs')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(os.listdir("../input"))


if __name__ == '__main__':
    lr = 1e-4
    epoch = 200

    model = nn.DataParallel(CatsDogsDenseNet()).to(DEVICE)

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
                        pred_class = 1 if pred_score > 0 else 0
                        if target_class != pred_class:
                            as_ = 'cat' if pred_class == CatsDogsDataset.LABEL_CAT else 'dog'
                            wavfile.write(f'mispred_{as_}_{pred_score:04f}.wav', 16000, samp.cpu().detach().numpy().T)
                            print(f'misprediction ({as_}): {pred_score:04f}')

        print(f'Test Accuracy: {sum(test_accs) / len(test_accs)}')

    # save module
    print('Saving model')
    torch.save(model.module.state_dict(), 'model.pt')
