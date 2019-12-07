import os
import random
import torch
from torch import nn, optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from scipy.io import wavfile
from .model import CatsDogsModel
from .dataset import BarkMeowDB

# Define some constants used throughout the training
AUDIO_DIR = './barkmeow'  # directory that contains 'cat' and 'dog' directories
DATA_OUT_ROOT = './data_out'
# directory where tensorboard logs are stored
LOG_DIR = os.path.join(DATA_OUT_ROOT, 'logs')
MODEL_PATH = os.path.join(DATA_OUT_ROOT, 'model.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 3e-4  # learning rate
EPOCH = 100
SAMPLE_RATE = 16000  # sample rate - used to save audio of mispredicted examples

# fix random seed to make the training deterministic
torch.manual_seed(8912)
np.random.seed(9912)

writer = SummaryWriter(log_dir=LOG_DIR)

model = nn.DataParallel(CatsDogsModel()).to(DEVICE)

# read dataset and create dataloaders
dataset = BarkMeowDB(AUDIO_DIR)
num_data = len(dataset)
num_train = int(num_data * 0.9)

all_indices = list(range(num_data))
random.shuffle(all_indices)

train_dataloader = data.DataLoader(
    dataset,
    batch_size=5,
    sampler=data.sampler.SubsetRandomSampler(all_indices[:num_train]),
    pin_memory=True,
    drop_last=True)
test_dataloader = data.DataLoader(
    dataset,
    batch_size=5,
    sampler=data.sampler.SubsetRandomSampler(all_indices[num_train:]),
    pin_memory=True,
    drop_last=True)

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)

# binary cross entropy loss
loss_func = nn.BCEWithLogitsLoss()

for i in range(EPOCH):
    losses = []
    accs = []
    for targets, samples in train_dataloader:
        targets = targets.type(torch.FloatTensor).to(DEVICE)
        samples = samples.to(DEVICE)

        out = torch.squeeze(model(samples))
        loss = loss_func(out, targets)

        out_bool = torch.sigmoid(out) > 0.5
        target_bool = targets > 0.5
        pred = (out_bool == target_bool).type(torch.FloatTensor)
        accuracy = pred.sum() / pred.numel()

        # accumulate accuracy and loss values for later calculating mean values of each
        accs.append(accuracy.data)
        losses.append(loss.data)

        # back-prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # log values
    avg_loss = sum(losses) / len(losses)
    avg_acc = sum(accs) / len(accs)
    writer.add_scalar('train/loss', avg_loss, global_step=i)
    writer.add_scalar('train/accuracy', avg_acc, global_step=i)
    print(f'epoch: {i}, loss: {avg_loss}, accuracy: {avg_acc}')

    test_losses = []
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

            test_accs.append(accuracy.data)
            test_losses.append(loss.data)

            # save any that are mispredicted at the end of training
            if i == EPOCH - 1:
                for target_class, samp, pred_score in zip(targets, samples, out):
                    pred_class = 1 if pred_score > 0 else 0
                    if target_class != pred_class:
                        # misclassified class name
                        as_ = dataset.class_label_name[pred_class]

                        fname = f'mispred_as_{as_}_{pred_score:04f}.wav'
                        mispred_path = os.path.join(DATA_OUT_ROOT, fname)
                        audio_data = samp.cpu().detach().numpy().T

                        # save misclassified audio
                        wavfile.write(mispred_path, SAMPLE_RATE, audio_data)
                        print(f'mispredicted (as: {as_}): {pred_score:04f}')

    avg_test_acc = sum(test_accs) / len(test_accs)
    avg_test_loss = sum(test_losses) / len(test_losses)
    print(f'TEST: {i}, loss:, {avg_test_loss}, accuracy: {avg_test_acc}')
    writer.add_scalar('test/loss', avg_test_loss, global_step=i)
    writer.add_scalar('test/accuracy', avg_test_acc, global_step=i)

# save module
print(f'Saving model: {MODEL_PATH}')
torch.save(model.module.state_dict(), MODEL_PATH)
