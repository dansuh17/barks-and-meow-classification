import os
import torch
from torch import optim
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from .model import CatsDogsModel


def load_model(model_file='model.pt'):
    """
    Loads a pre-trained model.

    Args:
        model_file (str): path to model file - usually has extension '.pt'

    Returns:
        model (torch.nn.Module): model instance in evaluation mode
    """
    model: torch.nn.Module = CatsDogsModel()
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()  # fix as evaluation mode
    return model


def auralize(model, is_cat: bool, out_path: str, n_iter=5000, samp_rate=16000, lr=0.01):
    """
    Find the best signal that would lead to highest probability to the provided class.

    Args:
        model (torch.nn.Module): pre-trained model instance
        is_cat (bool): ``True`` if you want to generate a cat sound, ``False`` for a dog sound
        out_path (str): output path for generated audio and images
        n_iter (int): number of iterations
        samp_rate (int): audio sample rate
        lr (float): learning rate

    Returns:
        auralized (torch.FloatTensor): generated audio signal
    """
    # create output directories if they don't exist
    img_out_dir = os.path.join(out_path, 'img')
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir, exist_ok=True)
        print(f'{img_out_dir} directory created')

    wav_out_dir = os.path.join(out_path, 'wav')
    if not os.path.exists(wav_out_dir):
        os.makedirs(wav_out_dir, exist_ok=True)
        print(f'{wav_out_dir} directory created')

    initial_signal = torch.rand((1, 1, samp_rate))  # 1-sec of random samples
    # allow parameter updates from back-propagation
    initial_signal.requires_grad = True
    optimizer = optim.SGD([initial_signal], lr=lr, weight_decay=0.05)

    target_name = 'cat' if is_cat else 'dog'

    # train the 'input' with respect to desired output target
    # i.e. find the best input that yields the desired target
    auralized = initial_signal
    for i in range(n_iter):
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
        print(f'loss: {class_loss.squeeze().data}, is_cat: {is_cat}')

        if i % 500 == 0:
            auralized_waveform = np.squeeze(auralized.detach().numpy())
            name = f'{target_name}_it{i}'

            # save the waveform image
            png_file = name + '.png'
            plt.figure()
            plt.plot(auralized_waveform)
            plt.savefig(os.path.join(img_out_dir, png_file))

            # save the example audio file
            wav_file = name + '.wav'
            wav_path = os.path.join(wav_out_dir, wav_file)  # full path
            wavfile.write(wav_path, samp_rate, auralized_waveform)

            # save spectrogram
            save_spectrogram(wav_path, out_path=img_out_dir)
    return auralized


def save_spectrogram(audio_path: str, out_path: str, n_fft=512, hop_length=128):
    """
    Saves the spectrogram of the provided audio file.

    Args:
        audio_path (str): path to audio file
        out_path (str): output directory
        n_fft (int): number of FFT kernels
        hop_length (int): hop size
    """
    audio_samp, _ = librosa.load(audio_path)
    plt.figure(figsize=(4, 8))
    mag_spec = np.abs(librosa.stft(audio_samp, n_fft=n_fft, hop_length=hop_length))
    D = librosa.amplitude_to_db(mag_spec, ref=np.max)
    plt.subplot(1, 1, 1)
    librosa.display.specshow(D, x_axis='time')
    plt.title(audio_path)

    base = os.path.basename(audio_path)
    rel, _ = os.path.splitext(base)

    plt.savefig(os.path.join(out_path, rel + '_spec.png'))


if __name__ == '__main__':
    pretrained_model_path = 'data_out/model.pt'
    model = load_model(pretrained_model_path)

    # fix random seed to make auralized examples deterministic
    torch.manual_seed(191)
    np.random.seed(83890)

    auralize(model=model, is_cat=True, out_path='data_out/auralized')
    auralize(model=model, is_cat=False, out_path='data_out/auralized')
