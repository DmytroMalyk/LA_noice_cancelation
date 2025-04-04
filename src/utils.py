import numpy as np

def stft(audio, fft_size=1024, hop_size=512):
    frames = []
    window = np.hanning(fft_size)
    for i in range(0, len(audio) - fft_size, hop_size):
        segment = audio[i:i + fft_size] * window
        spectrum = np.fft.fft(segment)
        frames.append(spectrum[:fft_size // 2 + 1])
    return np.array(frames).T

def svd_decompose(matrix, num_iter=1000, tol=1e-6):
    m, n = matrix.shape
    U = np.random.randn(m, n)
    Vt = np.random.randn(n, n)
    for _ in range(num_iter):
        Vt_new = np.dot(matrix.T, U)
        Vt_new = normalize_columns(Vt_new)
        U_new = np.dot(matrix, Vt_new.T)
        U_new = normalize_columns(U_new)
        if np.linalg.norm(U_new - U) < tol and np.linalg.norm(Vt_new - Vt) < tol:
            break
        U, Vt = U_new, Vt_new
    S = np.diag([np.linalg.norm(np.dot(matrix, v)) for v in Vt])
    return U, S, Vt

def normalize_columns(matrix):
    norms = np.linalg.norm(matrix, axis=0)
    return matrix / norms
