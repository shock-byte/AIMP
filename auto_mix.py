#!/usr/bin/env python3
"""
Professional Apple-style DJ Automix with spectral-preserving crossfade
- Micro-stretch overlap only
- STFT-based crossfade for full detail
- Energy-aware fade
- Optional smooth sidechain drums
"""

import os
import numpy as np
import librosa
import soundfile as sf
import yt_dlp

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
# Downloader
# -----------------------
def download_audio_from_query(query, out_dir="downloads"):
    """
    Search YouTube for the query, download the top result audio, and convert to WAV.
    Requires ffmpeg to be installed and available in PATH.
    Returns the path to the downloaded .wav file.
    """
    os.makedirs(out_dir, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(out_dir, '%(id)s.%(ext)s'),
        'noplaylist': True,
        'quiet': False,
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '0',
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch1:{query}", download=True)
        if isinstance(info, dict) and 'entries' in info and info['entries']:
            info = info['entries'][0]
        file_id = info.get('id') if isinstance(info, dict) else None
        if not file_id:
            raise RuntimeError("Failed to download audio for query: " + query)
        wav_path = os.path.join(out_dir, f"{file_id}.wav")
        if not os.path.exists(wav_path):
            # Some extractors may keep original extension if already wav
            # Fallback: try to locate any file with this id
            candidates = [p for p in os.listdir(out_dir) if p.startswith(file_id + ".")]
            for cand in candidates:
                if cand.endswith('.wav'):
                    wav_path = os.path.join(out_dir, cand)
                    break
        return wav_path

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

def automix_from_queries(query_a, query_b, out_file,
                         bars_overlap=8,
                         beats_per_bar=4,
                         drum_file=None,
                         drum_db=-12.0,
                         sidechain=True):
    """
    End-to-end: search+download two queries and automix them like Apple Music.
    """
    file_a = download_audio_from_query(query_a)
    file_b = download_audio_from_query(query_b)
    automix_pro(
        file_a=file_a,
        file_b=file_b,
        out_file=out_file,
        bars_overlap=bars_overlap,
        beats_per_bar=beats_per_bar,
        drum_file=drum_file,
        drum_db=drum_db,
        sidechain=sidechain,
    )

# -----------------------
# Run script
# -----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apple-style DJ Automix: local files or YouTube queries")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--files", nargs=2, metavar=("FILE_A", "FILE_B"), help="Two local audio files to blend")
    src.add_argument("--queries", nargs=2, metavar=("QUERY_A", "QUERY_B"), help="Two search queries to fetch from YouTube")

    parser.add_argument("--out", default="auto_mix.wav", help="Output WAV path (default: auto_mix.wav)")
    parser.add_argument("--bars-overlap", type=int, default=8, help="Number of bars to overlap (default: 8)")
    parser.add_argument("--beats-per-bar", type=int, default=4, help="Beats per bar for detection (default: 4)")
    parser.add_argument("--drum-file", default=None, help="Optional drum loop WAV/MP3 to underlay and sidechain against")
    parser.add_argument("--drum-db", type=float, default=-12.0, help="Drum loop level in dB (default: -12.0)")
    parser.add_argument("--no-sidechain", action="store_true", help="Disable smooth sidechain ducking")

    args = parser.parse_args()
    sidechain_enabled = not args.no_sidechain

    if args.files:
        file_a, file_b = args.files
        automix_pro(
            file_a=file_a,
            file_b=file_b,
            out_file=args.out,
            bars_overlap=args.bars_overlap,
            beats_per_bar=args.beats_per_bar,
            drum_file=args.drum_file,
            drum_db=args.drum_db,
            sidechain=sidechain_enabled,
        )
    else:
        query_a, query_b = args.queries
        automix_from_queries(
            query_a=query_a,
            query_b=query_b,
            out_file=args.out,
            bars_overlap=args.bars_overlap,
            beats_per_bar=args.beats_per_bar,
            drum_file=args.drum_file,
            drum_db=args.drum_db,
            sidechain=sidechain_enabled,
        )
