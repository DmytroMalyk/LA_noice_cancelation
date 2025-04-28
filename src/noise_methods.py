import numpy as np
import librosa
import colorednoise as cn

def add_white_noise(clean, noise_level=0.001):
    noise = noise_level * np.random.normal(0, 1, size=clean.shape)
    return clean + noise

def add_pink_noise(clean, noise_level=0.001, beta=1):
    noise = noise_level * cn.powerlaw_psd_gaussian(beta, len(clean))
    return clean + noise

def add_click_noise(clean, num_clicks=100, click_strength=0.5, click_length=10):
    noised = clean.copy()
    for _ in range(num_clicks):
        idx = np.random.randint(0, len(clean))
        end = min(len(clean), idx + click_length)
        noised[idx:end] += click_strength * np.random.randn(end - idx)
    return noised

# Example usage
if __name__ == "__main__":
    clean_path = "data/raw/sp13.wav"
    y, sr = librosa.load(clean_path, sr=None)

    white_noised = add_white_noise(y)
    pink_noised = add_pink_noise(y)
    click_noised = add_click_noise(y)

    # Save to disk for testing
    import soundfile as sf
    sf.write("data/processed/white_noise.wav", white_noised, sr)
    sf.write("data/processed/pink_noise.wav", pink_noised, sr)
    sf.write("data/processed/click_noise.wav", click_noised, sr)
