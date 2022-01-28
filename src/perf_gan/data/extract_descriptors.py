import librosa as li
from perf_gan.data.descriptors import extract_lo, extract_f0


def extract_f0_lo(filename, sr, block_size):
    audio, fs = li.load(filename, sr=sr)
    lo = extract_lo(audio, sr, block_size)
    f0 = extract_f0(audio, sr, block_size)

    return f0, lo


path = "data/audio/"
filename = "sample1.wav"

f0, lo = extract_f0_lo(path + filename, 16000, 100)
print(f"f0 {f0.shape}, lo {lo.shape}")
