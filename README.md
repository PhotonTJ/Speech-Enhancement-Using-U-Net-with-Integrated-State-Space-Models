# Speech-Enhancement-Using-U-Net-with-Integrated-State-Space-Models
# Audio U-Net LRU: Deep Learning Audio Denoising

A PyTorch implementation of a U-Net architecture enhanced with Linear Recurrent Units (LRU) for real-time audio denoising. This model combines the spatial understanding of U-Net with the temporal modeling capabilities of LRU blocks to effectively remove noise from audio signals.

## 🎯 Features

- **U-Net Architecture**: Encoder-decoder structure with skip connections optimized for 1D audio signals
- **LRU Integration**: Linear Recurrent Units in the bottleneck for superior temporal sequence modeling
- **Real-time Processing**: Efficient inference suitable for real-time audio applications
- **Synthetic Data Generation**: Built-in tools for creating training datasets
- **Flexible Input**: Handles various audio formats and lengths
- **GPU Acceleration**: CUDA support for faster training and inference

## 🏗️ Architecture

The model consists of three main components:

1. **Encoder**: Convolutional layers with downsampling to extract hierarchical features
2. **LRU Bottleneck**: Linear Recurrent Unit for temporal sequence modeling
3. **Decoder**: Upsampling layers with skip connections to reconstruct clean audio

### Key Components

- **LRUBlock**: Implements complex-valued linear recurrent processing
- **DownPool1D/UpPool1D**: Efficient 1D pooling operations for audio
- **UNetLRUAudio**: Main model combining U-Net structure with LRU

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchaudio librosa numpy pathlib
```

### Basic Usage

```python
import torch
from audio_unet_lru import UNetLRUAudio

# Initialize model
model = UNetLRUAudio(in_channels=1, out_channels=1)

# Load your noisy audio (shape: [batch_size, channels, length])
noisy_audio = torch.randn(1, 1, 16000)  # 1 second at 16kHz

# Denoise
with torch.no_grad():
    clean_audio = model(noisy_audio)

# Or use the convenience method for single samples
denoised = model.denoise_single(noisy_audio.squeeze())
```

## 📊 Training

### Using Synthetic Data

```python
from audio_unet_lru import create_synthetic_training_data, train_audio_unet_lru

# Generate synthetic training data
noisy_dir, clean_dir = create_synthetic_training_data(
    "./synthetic_audio_data", 
    num_samples=1000
)

# Train the model
model = train_audio_unet_lru(
    noisy_dir=noisy_dir,
    clean_dir=clean_dir,
    epochs=50,
    batch_size=16,
    learning_rate=1e-3
)
```

### Using Custom Data

```python
from audio_unet_lru import AudioDenoisingDataset, train_audio_unet_lru

# Prepare your data directories
# noisy_dir/ should contain: noisy_001.wav, noisy_002.wav, ...
# clean_dir/ should contain: clean_001.wav, clean_002.wav, ...

model = train_audio_unet_lru(
    noisy_dir="path/to/noisy/audio",
    clean_dir="path/to/clean/audio",
    epochs=100,
    batch_size=32
)
```

## 💾 Model Checkpoints

The training script automatically saves checkpoints:

```python
# Load a saved model
model = UNetLRUAudio()
checkpoint = torch.load('audio_unet_lru_final.pt')
model.load_state_dict(checkpoint)
model.eval()
```

## 🔧 Configuration

### Model Parameters

```python
model = UNetLRUAudio(
    in_channels=1,      # Input channels (1 for mono audio)
    out_channels=1,     # Output channels (1 for mono audio)
    features=[16, 32, 64, 128]  # Feature dimensions for each layer
)
```

### Training Parameters

```python
train_audio_unet_lru(
    noisy_dir="path/to/noisy",
    clean_dir="path/to/clean",
    epochs=50,          # Number of training epochs
    batch_size=16,      # Batch size
    learning_rate=1e-3, # Learning rate
)
```

## 📈 Performance

The model is designed for:
- **Sample Rate**: 16kHz (configurable)
- **Input Length**: Variable (default 16000 samples = 1 second)
- **Real-time Factor**: < 0.1 on modern GPUs
- **Memory Usage**: ~500MB GPU memory for inference

## 🛠️ Advanced Usage

### Batch Processing

```python
# Process multiple audio files
noisy_samples = [audio1, audio2, audio3]  # List of torch tensors
denoised_samples = model.denoise_multiple(noisy_samples)
```

### Custom Loss Functions

```python
import torch.nn as nn

# Example: Spectral loss
def spectral_loss(output, target):
    stft_output = torch.stft(output, n_fft=512, return_complex=True)
    stft_target = torch.stft(target, n_fft=512, return_complex=True)
    return nn.MSELoss()(torch.abs(stft_output), torch.abs(stft_target))
```

## 📁 Project Structure

```
audio_unet_lru/
├── audio_unet_lru.py          # Main implementation
├── synthetic_audio_data/       # Generated training data
│   ├── noisy/                 # Noisy audio files
│   └── clean/                 # Clean audio files
├── checkpoints/               # Model checkpoints
├── test_denoised.wav          # Example output
└── README.md                  # This file
```

## 🔬 Model Architecture Details

### LRU Block

The Linear Recurrent Unit processes sequences using:
- Complex-valued state transitions
- Learnable eigenvalues (λ) parameterized in log space
- Efficient FFT-based convolution for long sequences

### U-Net Modifications

- **1D Convolutions**: Adapted for audio signals
- **Skip Connections**: Preserve high-frequency details
- **Adaptive Pooling**: Handles variable-length inputs

## 🎵 Audio Processing Pipeline

1. **Preprocessing**: Load and normalize audio to [-1, 1]
2. **Segmentation**: Split long audio into manageable chunks
3. **Denoising**: Process through U-Net LRU model
4. **Reconstruction**: Combine processed segments
5. **Postprocessing**: Apply optional filtering and normalization

## 🧪 Evaluation

### Metrics

The model can be evaluated using:
- **Signal-to-Noise Ratio (SNR)**
- **Perceptual Evaluation of Speech Quality (PESQ)**
- **Short-Time Objective Intelligibility (STOI)**

### Benchmarking

```python
import librosa
import numpy as np

def calculate_snr(clean, noisy):
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - noisy) ** 2)
    return 10 * np.log10(signal_power / noise_power)

# Example evaluation
clean_audio, _ = librosa.load('clean.wav', sr=16000)
noisy_audio, _ = librosa.load('noisy.wav', sr=16000)
denoised_audio = model.denoise_single(torch.tensor(noisy_audio))

snr_before = calculate_snr(clean_audio, noisy_audio)
snr_after = calculate_snr(clean_audio, denoised_audio.numpy())
print(f"SNR improvement: {snr_after - snr_before:.2f} dB")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Linear Recurrent Units (LRU) implementation based on research from [Orvieto et al., 2023]
- U-Net architecture inspired by [Ronneberger et al., 2015]
- Audio processing utilities built with librosa and torchaudio

## 📧 Contact

For questions and support, please open an issue or contact [your-email@example.com].

---

**Note**: This implementation is for research and educational purposes. For production use, consider additional optimizations and testing.
