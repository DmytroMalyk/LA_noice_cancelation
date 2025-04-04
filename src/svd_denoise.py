import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from utils import svd, stft, istft

def plot_spectrogram(S, sr, title='Spectrogram', filename=None):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr, y_axis='log', x_axis='time', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def svd_denoise(audio_path, output_path, rank=10):
    y, sr = librosa.load(audio_path, sr=None)

    S_complex = stft(y)
    magnitude, phase = np.abs(S_complex), np.angle(S_complex)

    plot_spectrogram(magnitude, sr, title='Original Spectrogram', filename='original_spectrogram.png')
    U, s, Vt = svd(magnitude)

    s_denoised = np.zeros_like(s)
    s_denoised[:rank] = s[:rank]

    denoised_magnitude = np.dot(U, np.dot(np.diag(s_denoised), Vt))

    plot_spectrogram(denoised_magnitude, sr, title='Denoised Spectrogram', filename='denoised_spectrogram.png')
    S_denoised_complex = denoised_magnitude * np.exp(1j * phase)

    y_denoised = istft(S_denoised_complex)
    sf.write(output_path, y_denoised, sr)

if __name__ == "__main__":
    input_file = "data/raw/sp13_restaurant_sn15.wav"
    output_file = "data/processed/denoised_audio.wav"
    svd_denoise(input_file, output_file)
