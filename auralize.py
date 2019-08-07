import torch
from torch import optim
import sys
from cats_and_dogs import CatsDogsModel3
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io import wavfile


if __name__ == '__main__':
    # model = CatsDogsModel3()
    # model.load_state_dict(torch.load('model.pt', map_location='cpu'))
    # model.eval()

    # initial_image = torch.rand((1, 1, 16000))
    # initial_image.requires_grad = True
    # optimizer = optim.SGD([initial_image], lr=5, weight_decay=0.05)

    # for i in range(200):
    #     out = model(initial_image)
    #     class_loss = out  # cat

    #     optimizer.zero_grad()
    #     class_loss.backward()
    #     optimizer.step()
    #     print(class_loss)

    #     if i % 20 == 0:
    #         plt.figure()
    #         plt.plot(np.squeeze(initial_image.detach().numpy()))
    #         plt.show()
    #         print(i)
    #         wavfile.write(f'cat_{i}.wav', 16000, np.squeeze(initial_image.detach().numpy()))

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
