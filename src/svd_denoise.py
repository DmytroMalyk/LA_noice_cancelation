import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from utils import svd, stft, istft, compute_mse, compute_stoi
from noise_methods import add_white_noise, add_pink_noise, add_click_noise

def plot_spectrogram(S, sr, title='Spectrogram', filename=None):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(S), ref=np.max), sr=sr, y_axis='log', x_axis='time', cmap='inferno')
    plt.title(title)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def svd_denoise_from_clean(clean_audio_path, noisy_output_path, denoised_output_path, noise_method='add_white_noise', snr_db=20, threshold=0.9):
    # Load clean audio
    y_clean, sr = librosa.load(clean_audio_path, sr=None)

    # Select noise method
    if noise_method == 'add_white_noise':
        y_noisy = add_white_noise(y_clean, noise_level=10**(-snr_db / 20))
    elif noise_method == 'add_pink_noise':
        y_noisy = add_pink_noise(y_clean, noise_level=10**(-snr_db / 20))
    elif noise_method == 'add_click_noise':
        y_noisy = add_click_noise(y_clean, num_clicks=100, click_strength=0.5, click_length=10)
    else:
        raise ValueError("Invalid noise method selected")

    # Save noised version
    sf.write(noisy_output_path, y_noisy, sr)

    # STFT
    S_complex = stft(y_noisy)
    magnitude, phase = np.abs(S_complex), np.angle(S_complex)

    # Normalize magnitude
    max_mag = np.max(magnitude)
    magnitude_norm = magnitude / max_mag

    # Plot original clean, noised and normalized spectrograms
    plot_spectrogram(stft(y_clean), sr, title='Clean Spectrogram', filename='clean_spectrogram.png')
    plot_spectrogram(magnitude, sr, title='Noised Spectrogram', filename='noised_spectrogram.png')

    # SVD
    U, s, Vt = svd(magnitude_norm)

    # Denoising by thresholding
    energy = np.sum(s)
    target_energy = threshold * energy

    cumulative_energy = np.cumsum(s)
    k = np.searchsorted(cumulative_energy, target_energy) + 1

    s_denoised = np.zeros_like(s)
    s_denoised[:k] = s[:k]

    denoised_magnitude_norm = np.dot(U, np.dot(np.diag(s_denoised), Vt))

    # Denormalize
    denoised_magnitude = denoised_magnitude_norm * max_mag

    # Plot denoised spectrogram
    plot_spectrogram(denoised_magnitude, sr, title='Denoised Spectrogram', filename='denoised_spectrogram.png')

    # ISTFT
    S_denoised_complex = denoised_magnitude * np.exp(1j * phase)
    y_denoised = istft(S_denoised_complex)

    # Save denoised audio
    sf.write(denoised_output_path, y_denoised, sr)

    # Evaluation Metrics
    min_len = min(len(y_clean), len(y_denoised))
    y_clean_eval = y_clean[:min_len]
    y_denoised_eval = y_denoised[:min_len]

    mse_value = compute_mse(y_clean_eval, y_denoised_eval)
    stoi_value = compute_stoi(y_clean_eval, y_denoised_eval, sr)

    print(f"\n Evaluation Metrics:")
    print(f"   MSE  = {mse_value:.6f}")
    print(f"   STOI = {stoi_value:.4f}")


if __name__ == "__main__":
    clean_file = "data/raw/sp02.wav"
    noised_file = "data/processed/noised_audio.wav"
    denoised_file = "data/processed/denoised_audio.wav"

    svd_denoise_from_clean(clean_file, noised_file, denoised_file, noise_method='add_white_noise', snr_db=60, threshold=0.9)