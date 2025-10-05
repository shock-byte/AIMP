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

Install Python deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Place or reference your audio files (WAV/MP3). Then run:

```bash
python3 auto_mix.py
```

Edit the call at the bottom of `auto_mix.py` to point to your files or import and call the function in your own script.

Output mix will be written to `auto_mix_pro.wav` or `auto_mix_final.wav` depending on the function used.

## Notes
- If `gh` (GitHub CLI) is installed, you can create and push a repo with one command (see below).
- Large audio assets are not committed by default; generated outputs matching `auto_mix*.wav`/`*.mp3` are ignored.
