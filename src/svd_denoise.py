import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from utils import stft, svd_decompose

def plot_spectrogram(spectrogram, title='Spectrogram', filename=None):
    """
    Function to plot the magnitude of the spectrogram and save it to a file.
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(np.log1p(np.abs(spectrogram)), aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(label='Log Magnitude')
    plt.title(title)
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')

    if filename:
        plt.savefig(filename)  # Save the plot as an image file
    else:
        plt.show()  # Show the plot if no filename is provided

def svd_denoise(audio_path, output_path, threshold_ratio=0.1):
    y, sr = sf.read(audio_path)

    # Perform the STFT and get the full complex spectrogram
    spectrogram = stft(y)

    # Get magnitude and phase from the complex spectrogram
    magnitude, phase = np.abs(spectrogram), np.angle(spectrogram)

    # Save the original spectrogram
    plot_spectrogram(spectrogram, title='Original Spectrogram', filename='original_spectrogram.png')

    # Perform SVD on the magnitude spectrogram
    U, S, Vt = svd_decompose(magnitude)

    # Apply threshold to singular values
    threshold = threshold_ratio * np.max(S)
    S_denoised = np.where(S > threshold, S, 0)

    # Reconstruct the denoised magnitude spectrogram
    denoised_magnitude = np.dot(U, np.dot(np.diag(S_denoised), Vt))

    # Recreate the denoised spectrogram with the same phase shape as magnitude
    denoised_spectrogram = denoised_magnitude * np.exp(1j * phase)

    # Save the denoised spectrogram
    plot_spectrogram(denoised_spectrogram, title='Denoised Spectrogram', filename='denoised_spectrogram.png')

    # Convert back to time-domain signal using librosa's inverse STFT
    y_denoised = librosa.istft(denoised_spectrogram)

    # Save the denoised audio to the specified output file
    sf.write(output_path, y_denoised, sr)

if __name__ == "__main__":
    input_file = "data/raw/car_15dB/sp06_car_sn15.wav"
    output_file = "data/processed/denoised_audio.wav"
    svd_denoise(input_file, output_file)
