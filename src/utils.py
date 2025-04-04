import numpy as np
import matplotlib.pyplot as plt

def stft(audio, fft_size=1024, hop_size=512):
    window = np.hanning(fft_size)
    frames = []
    for i in range(0, len(audio) - fft_size, hop_size):
        frame = audio[i:i+fft_size] * window
        spectrum = np.fft.fft(frame)[:fft_size//2+1]
        frames.append(spectrum)
    return np.array(frames).T

def istft(S_complex, fft_size=1024, hop_size=512):
    time_frames = S_complex.shape[1]
    signal_len = time_frames * hop_size + fft_size
    signal = np.zeros(signal_len)
    window = np.hanning(fft_size)
    for n, i in enumerate(range(0, len(signal) - fft_size, hop_size)):
        spectrum = np.zeros(fft_size, dtype=np.complex64)
        spectrum[:fft_size//2+1] = S_complex[:, n]
        spectrum[fft_size//2+1:] = np.conj(S_complex[1:fft_size//2][::-1, n])
        frame = np.fft.ifft(spectrum).real * window
        signal[i:i+fft_size] += frame
    return signal

def plot_spectrogram(S, sr, title='Spectrogram', filename=None):
    plt.figure(figsize=(10, 6))
    plt.imshow(20 * np.log10(np.abs(S) + 1e-6), aspect='auto', origin='lower', cmap='inferno',
               extent=[0, S.shape[1], 0, sr/2])
    plt.colorbar(label='Magnitude (dB)')
    plt.title(title)
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency (Hz)')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def power_iteration(A, num_simulations=100):
    b = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        b_k1 = np.dot(A, b)

        b_k1_norm = np.linalg.norm(b_k1)
        b = b_k1 / b_k1_norm

    eigenvalue = np.dot(np.dot(A, b), b) / np.dot(b, b)

    return eigenvalue, b


def power_iteration_deflation(A, num_simulations=100):
    n = A.shape[0]
    eig_vals = np.zeros(n)
    eig_vecs = np.zeros((n, n))

    for i in range(n):
        eig_val, eig_vec = power_iteration(A, num_simulations)

        eig_vals[i] = eig_val
        eig_vecs[:, i] = eig_vec

        A = A - eig_val * np.outer(eig_vec, eig_vec)

    return eig_vals, eig_vecs


def svd(A):
    AT = A.T
    ATA = AT.dot(A)
    
    eig_vals, eig_vecs = power_iteration_deflation(ATA)

    sorted_indices = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sorted_indices]
    eig_vecs = eig_vecs[:, sorted_indices]

    s = np.sqrt(eig_vals)

    V = eig_vecs
    U = A.dot(V) / s

    return U, s, V.T
