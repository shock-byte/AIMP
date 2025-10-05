#!/usr/bin/env python3
"""
Professional Apple-style DJ Automix with spectral-preserving crossfade
- Micro-stretch overlap only
- STFT-based crossfade for full detail
- Energy-aware fade
- Optional smooth sidechain drums
"""

import numpy as np
import librosa
import soundfile as sf

# -----------------------
# Helpers
# -----------------------
def load_audio(file):
    y, sr = librosa.load(file, sr=None, mono=True)
    print(f"Loaded {file}: {len(y)/sr:.2f}s at {sr}Hz")
    return y, sr

def detect_beats_and_bars(y, sr, beats_per_bar=4):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)
    bar_times = beat_times[::beats_per_bar]
    return tempo, beats, bar_times

def stft_crossfade(seg_a, seg_b):
    """
    Spectral-preserving S-curve crossfade using STFT
    """
    n_fft = 2048
    hop_length = 512
    A = librosa.stft(seg_a, n_fft=n_fft, hop_length=hop_length)
    B = librosa.stft(seg_b, n_fft=n_fft, hop_length=hop_length)
    min_cols = min(A.shape[1], B.shape[1])
    A = A[:, :min_cols]
    B = B[:, :min_cols]

    # S-curve fade
    t = np.linspace(-np.pi/2, np.pi/2, min_cols)
    fade_in = (1 + np.sin(t)) / 2.0
    fade_out = 1.0 - fade_in

    cross = A * fade_out + B * fade_in
    return librosa.istft(cross, hop_length=hop_length, length=min_cols*hop_length)

def sidechain_smooth(track, drum, attack=0.01, release=0.1, sr=44100, ratio=0.5):
    """
    Smooth sidechain: track volume ducked by drum with attack/release envelope
    """
    env = np.zeros_like(track)
    drum_env = np.abs(drum)
    alpha_a = np.exp(-1/(attack*sr))
    alpha_r = np.exp(-1/(release*sr))
    for i in range(1, len(track)):
        if drum_env[i] > env[i-1]:
            env[i] = alpha_a * env[i-1] + (1-alpha_a) * drum_env[i]
        else:
            env[i] = alpha_r * env[i-1] + (1-alpha_r) * drum_env[i]
    gain = 1.0 - env * ratio
    return track * gain

# -----------------------
# Automix Function
# -----------------------
def automix_pro(file_a, file_b, out_file,
                bars_overlap=8,
                beats_per_bar=4,
                drum_file=None,
                drum_db=-12.0,
                sidechain=True):
    # Load tracks
    y_a, sr_a = load_audio(file_a)
    y_b, sr_b = load_audio(file_b)
    sr = sr_a
    if sr_a != sr_b:
        y_b = librosa.resample(y_b, orig_sr=sr_b, target_sr=sr_a)

    # Detect tempo & bars
    tempo_a, beats_a, bars_a = detect_beats_and_bars(y_a, sr, beats_per_bar)
    tempo_b, beats_b, bars_b = detect_beats_and_bars(y_b, sr, beats_per_bar)
    print(f"Tempo A: {tempo_a:.2f}, Tempo B: {tempo_b:.2f}")

    # -------------------
    # Determine overlap
    start_idx_a = max(0, len(bars_a) - bars_overlap - 1)
    start_idx_b = 0
    overlap_start_a = bars_a[start_idx_a]
    overlap_end_a = bars_a[start_idx_a + bars_overlap] if start_idx_a + bars_overlap < len(bars_a) else bars_a[-1]
    overlap_len = overlap_end_a - overlap_start_a
    overlap_samples = int(overlap_len * sr)

    # Extract overlap segments
    seg_a = y_a[int(overlap_start_a*sr):int(overlap_start_a*sr)+overlap_samples]
    seg_b = y_b[int(bars_b[start_idx_b]*sr):int(bars_b[start_idx_b]*sr)+overlap_samples]

    # -------------------
    # Micro-stretch seg_b to match tempo of seg_a
    ratio = tempo_a / tempo_b
    max_adjust = 0.05  # ±5%
    if ratio > 1 + max_adjust:
        ratio = 1 + max_adjust
    elif ratio < 1 - max_adjust:
        ratio = 1 - max_adjust
    if abs(ratio - 1.0) > 0.001:
        seg_b = librosa.effects.time_stretch(seg_b, rate=ratio)

    # -------------------
    # Drum overlay
    drum_mix = None
    if drum_file:
        drum_y, drum_sr = load_audio(drum_file)
        if drum_sr != sr:
            drum_y = librosa.resample(drum_y, orig_sr=drum_sr, target_sr=sr)
        drum_y = np.tile(drum_y, int(np.ceil(overlap_samples / len(drum_y))))[:overlap_samples]
        drum_mix = drum_y * (10**(drum_db/20.0))

    # -------------------
    # Apply smooth sidechain
    if sidechain and drum_mix is not None:
        seg_a = sidechain_smooth(seg_a, drum_mix, sr=sr)
        seg_b = sidechain_smooth(seg_b, drum_mix, sr=sr)

    # -------------------
    # Spectral-preserving S-curve crossfade
    overlap = stft_crossfade(seg_a, seg_b)

    # Fix length mismatch between overlap and drum
    if drum_mix is not None:
        if len(drum_mix) < len(overlap):
            drum_mix = np.pad(drum_mix, (0, len(overlap)-len(drum_mix)))
        elif len(drum_mix) > len(overlap):
            drum_mix = drum_mix[:len(overlap)]
        overlap += drum_mix

    # -------------------
    # Assemble final track
    pre = y_a[:int(overlap_start_a*sr)]
    post = y_b[int(bars_b[start_idx_b]*sr)+overlap_samples:]
    final = np.concatenate([pre, overlap, post])

    # Normalize
    peak = np.max(np.abs(final))
    if peak > 1.0:
        final /= peak

    sf.write(out_file, final, sr)
    print(f"✅ Automix complete: {out_file}, overlap {overlap_len:.2f}s")

# -----------------------
# Run script
# -----------------------
if __name__ == "__main__":
    automix_pro(
        file_a="Thodi Si Daaru.wav",
        file_b="Sahiba.wav",
        out_file="auto_mix_pro.wav",
        bars_overlap=8,
        beats_per_bar=4,
        drum_file="IVS.wav",
        drum_db=-12.0,
        sidechain=True
    )
