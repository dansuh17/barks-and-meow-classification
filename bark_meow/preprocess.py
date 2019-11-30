import os
from scipy.io import wavfile
import librosa
import numpy as np


def preprocess_audio(root_dir: str, out_dir: str,
                     target_sr=16000, chunk_size=16000, hop_size=8000, silence_threshold_rms=0.01):
    """
    Preprocess audio files in one directory and produce a new dataset in another.
    Resamples to provided target sample rate and
    splits into chunks of provided chunk size.

    :param str root_dir: root directory of the original audio data
    :param str out_dir: output directory
    :param int target_sr: target sample rate
    :param int chunk_size: chunk length in samples
    :param int hop_size: hop size in number of samples
    :param float silence_threshold_rms: RMS threshold for silence detection
    """
    if not os.path.exists(out_dir):
        print(f'Creating directory: {out_dir}')
        os.mkdir(out_dir)

    for root, _, files in os.walk(root_dir):
        rel_path = os.path.relpath(root, root_dir)
        target_dir = os.path.join(out_dir, rel_path)
        # create target directory if target directory does not exist
        if not os.path.exists(target_dir):
            print(f'Creating directory: {target_dir}')
            os.mkdir(target_dir)

        for audio_f in files:
            full_path = os.path.join(root, audio_f)
            name, _ = os.path.splitext(audio_f)

            # load samples, resampled to target sample rate
            audio_data, _ = librosa.load(full_path, sr=target_sr)
            num_samples = len(audio_data)

            for i, start_idx in enumerate(range(0, num_samples, hop_size)):
                end_idx = start_idx + chunk_size
                # ignore if there are not enough samples
                if end_idx >= num_samples:
                    break

                samples_chunk = audio_data[start_idx:end_idx]

                rms = np.sqrt(np.mean(samples_chunk ** 2))
                # ignore silent data
                if rms < silence_threshold_rms:
                    continue

                # determine the destination file name
                dst_path = os.path.join(target_dir, f'{name}-{i}.wav')  # append the chunk index

                print(f'Writing: {dst_path}')
                wavfile.write(filename=dst_path, rate=target_sr, data=samples_chunk)


if __name__ == '__main__':
    preprocess_audio('../datasets/barkmeow', './barkmeow_db')
