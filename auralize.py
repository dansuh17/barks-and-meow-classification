import torch
from torch import optim
import sys
from bark_meow.train import CatsDogsModel3
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io import wavfile


def auralize(model_file='model.pt', iterate=200, is_cat=True):
    model = CatsDogsModel3()
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()

    initial_image = torch.rand((1, 1, 16000))  # 1-sec of random samples
    # allow parameter updates from back-propagation
    initial_image.requires_grad = True
    optimizer = optim.SGD([initial_image], lr=5, weight_decay=0.05)

    target_name = 'cat' if is_cat else 'dog'
    auralized = initial_image
    # train the 'input' with respect to desired output target
    # i.e. find the best input that yields the desired target
    for i in range(iterate):
        out = model(auralized)
        if is_cat:
            # we want the output to be close to 0 (cat label)
            class_loss = out
        else:
            # we want the output to be close to 1 (dog label)
            class_loss = (1 - out)

        # back-prop to input
        optimizer.zero_grad()
        class_loss.backward()
        optimizer.step()
        print(f'loss: {class_loss}, is_cat: {is_cat}')

        if i % 20 == 0:
            plt.figure()
            plt.plot(np.squeeze(initial_image.detach().numpy()))
            plt.show()
            wavfile.write(f'{target_name}_{i}.wav', 16000, np.squeeze(auralized.detach().numpy()))
    return auralized


if __name__ == '__main__':
    cat_samp, _ = librosa.load('cat_180.wav')
    plt.figure(figsize=(4, 8))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(cat_samp)), ref=np.max)
    plt.subplot(1, 1, 1)
    librosa.display.specshow(D, x_axis='time', y_axis='log')
    plt.title('cat')
    plt.show()

    dog_samp, _ = librosa.load('dog_180.wav')
    plt.figure(figsize=(4, 8))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(dog_samp)), ref=np.max)
    plt.subplot(1, 1, 1)
    librosa.display.specshow(D, x_axis='time', y_axis='log')
    plt.title('dog')
    plt.show()

    sys.exit(1)

    params = list(model.parameters())
    num_params = len(params)

    print(params[2].size())
    sys.exit(1)

    print(np.squeeze(params[1].detach().numpy()))

    for p in np.squeeze(params[1].detach().numpy()):
        plt.figure()
        plt.imshow(p)
        plt.show()

    # for i, p in enumerate(filter(lambda p: len(p.size()) <= 2, params)):
    #     plt.figure()
    #     plt.plot(p.detach().numpy())
    #     plt.show()
