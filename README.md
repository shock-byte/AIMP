# AIMP: Apple‑style DJ Automix

Professional automix script with spectral‑preserving crossfade, micro‑stretch, and optional drum underlay.

## Features
- S‑curve crossfade that preserves spectral detail
- Micro time‑stretch (±5–15%) to align tempos safely
- Beat/bar detection and bar‑aligned overlap
- Optional smooth sidechain ducking against a drum loop

## Requirements
- macOS or Linux recommended
- Python 3.9+
- System library: libsndfile (required by soundfile)
  - macOS: `brew install libsndfile`
  - Ubuntu/Debian: `sudo apt-get install libsndfile1`
- ffmpeg (required by yt-dlp for audio extraction)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`

Install Python deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Run via CLI with either local files or YouTube queries:

```bash
# Local files
python3 auto_mix.py --files "Song A.wav" "Song B.wav" --out mix.wav --bars-overlap 8 --beats-per-bar 4 --drum-file IVS.wav --drum-db -12

# YouTube queries (requires ffmpeg)
python3 auto_mix.py --queries "artist1 song1" "artist2 song2" --out ai_mix.wav
```

Programmatic usage:

```python
from auto_mix import automix_pro, automix_from_queries
automix_pro("a.wav", "b.wav", out_file="mix.wav")
automix_from_queries("query A", "query B", out_file="mix.wav")
```

Downloads are stored under `downloads/` (git-ignored). Ensure `ffmpeg` is installed.

## Notes
- If `gh` (GitHub CLI) is installed, you can create and push a repo with one command (see below).
- Large audio assets are not committed by default; generated outputs matching `auto_mix*.wav`/`*.mp3` are ignored.
