# LA_noice_cancelation

## 👨‍💻 Authors:
- **Dmytro Malyk** - (https://github.com/DmytroMalyk)
- **Katya Hushchuk** - (https://github.com/KatiaHushchuk)
- **Sofia Sydorchuk** - (https://github.com/SydorchukSofia)

This project applies audio denoising techniques using Singular Value Decomposition (SVD). It allows you to inject various types of noise (white, pink, click) into clean audio files and then denoise them using a custom SVD-based approach. The processed audio and visualizations (spectrograms) help evaluate the effectiveness of the denoising method.

---

## 📁 Project Structure

├── data/<br>
│   ├── raw/             # Original clean audio files<br>
│   └── processed/       # Noised and denoised output files<br>
├── src/<br>
│   ├── svd_denoise.py   # Main denoising pipeline using SVD<br>
│   ├── noise_methods.py # Noise generation methods<br>
│   └── utils.py         # STFT/ISTFT, SVD implementation, and evaluation metrics<br>
└── README.md


## 🚀 Features

- **Noise Generation**: Add white, pink, or click noise to clean audio.
- **STFT/ISTFT**: Custom implementation for time-frequency transformation.
- **SVD from Scratch**: Decompose and reconstruct magnitude spectrograms using eigenvalue decomposition.
- **Threshold-based Denoising**: Keep top singular values to suppress noise.
- **Metrics**: Evaluate performance using MSE and STOI (Short-Time Objective Intelligibility).
- **Visualization**: Save and plot spectrograms for clean, noisy, and denoised audio.

---

## 🧪 Example Usage

Run the denoising pipeline from `src/svd_denoise.py`:

```bash
python src/svd_denoise.py
```

By default, it uses:
- **Input file:** data/raw/sp02.wav
- **Output files:** data/processed/noised_audio.wav and data/processed/denoised_audio.wav
- **Noise type:** White noise
- **SVD energy threshold:** 0.97

## 🛠 Requirements

Install dependencies using pip:

```bash
pip install numpy matplotlib librosa soundfile pystoi
```

## 📝 Notes
- The SVD is computed from scratch using power iteration and deflation (no np.linalg.svd).
- This project is ideal for educational purposes to understand the role of SVD in signal processing.

## 📷 Sample Outputs

Spectrograms will be saved as PNGs:

- clean_spectrogram.png
- noised_spectrogram.png
- denoised_spectrogram.png
