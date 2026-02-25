# Particle Field Audio Visualizer

An audio-reactive particle field renderer with CRT phosphor glow and fisheye lens distortion. Particles are born at the centre of the frame, drift outward in slow spirals, and fade to nothing before they reach the edge — giving the impression of an infinite, dark field that breathes with the music.

Built as a sister project to [`STRING-VISUALIZER`]([https://github.com/](https://github.com/AmoghShet/STRING-VISUALIZER.git)) & [`OSCILLOSCOPE-VISUALIZER`](https://github.com/AmoghShet/OSCILLOSCOPE-VISUALIZER.git), sharing the same rendering pipeline and visual aesthetic; was used to create the music visualizer for [`Mo - Drowning (Sound Sketch) [Official Visualizer]`](https://www.youtube.com/watch?v=6HTqKCGnfRg)

---

## Preview

> *Rendered for "Drowning (Sound Sketch)" — F♯ minor, 87 BPM, atmospheric indie folk.*

The default color palette (`180 200 255` BGR) renders as a cool pale blue-white — like bioluminescent particles suspended in dark water.

---

## How It Works

### Particle Lifecycle

Every particle is born near the centre of the frame with a randomly assigned outward angle. It drifts along that angle for its entire lifetime, slowly rotating due to the spiral rate. As it moves further from the centre, its brightness follows a **bell curve**: dim at birth, rising to full brightness around 35% of the way out, then fading smoothly to zero before it ever reaches the screen edge. When a particle fades out, it silently respawns at the centre with a new random angle.

Because particles are staggered across the full radius on initialisation, the field is always populated — there is no cold start, and no visible boundary.

### Variable Viscosity

Damping is not uniform across the field. It follows a parabola that peaks at mid-radius (`4t(1-t)` where `t` is normalised distance from centre). Particles near the centre and near the edge experience less drag and move slightly faster; particles in the middle band move through the thickest part of the fluid. The effect is subtle but gives the motion an organic, layered quality — like moving through water of uneven density.

### Audio Reactivity

Each frame, four audio properties are extracted and mapped to distinct physics behaviours:

| Band | Frequency | Effect |
|------|-----------|--------|
| Low (< 300 Hz) | Sub-bass, cello stabs, kick | Radial outward burst — a pressure shockwave |
| Mid (300 Hz – 3 kHz) | Guitar body, vocals | Perpendicular swirl — a slow underwater current |
| High (> 3 kHz) | String noise, reverb air | Micro shimmer — surface light flickering |
| Onset strength | Transient detector | Additional scatter burst on plucks and attacks |

Velocity decays back toward each particle's personal drift after every burst, with speed controlled by the damping curve.

### CRT Glow

Each frame is composited from three layers:
- **Wide soft halo** — broad Gaussian blur at `2.5× sigma`, 35% opacity
- **Medium bloom** — tighter Gaussian at `1× sigma`, 60% opacity  
- **Core** — the raw particle dots at full brightness

All three are additively blended onto a persistent trail canvas, which itself fades by `trail_decay` each frame — giving particles a phosphorescent wake.

### Fisheye

A barrel distortion remap is pre-computed once at the start of the render and applied to a copy of each output frame. It never feeds back into the trail canvas, so distortion does not compound across frames.

---

## Requirements

```bash
pip install librosa numpy opencv-python scipy tqdm
```

FFmpeg must also be installed and available on your `PATH` (used for the final audio mux):

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

---

## Usage

### Quick preview (first 10 seconds)

```bash
python particle_visualizer.py --input track.wav --output preview.mp4 --preview
```

### Full render

```bash
python particle_visualizer.py \
  --input track.wav \
  --output visualizer.mp4 \
  --fps 60 \
  --n_particles 2000 \
  --line_color 180 200 255 \
  --drift_speed 0.4 \
  --damping 0.95 \
  --damping_edge 0.88 \
  --scatter_strength 6.0 \
  --trail_decay 0.88 \
  --fisheye_strength 0.5 \
  --centre_brightness 0.15 \
  --spiral_rate 0.003
```

### Recommended settings for slow/atmospheric tracks

```bash
python particle_visualizer.py \
  --input Drowning.wav --output drowning.mp4 \
  --fps 60 --n_particles 2000 \
  --line_color 180 200 255 \
  --drift_speed 0.4 --damping 0.95 --damping_edge 0.88 \
  --scatter_strength 5.0 --trail_decay 0.90 \
  --fisheye_strength 0.5 --centre_brightness 0.10 \
  --spiral_rate 0.003
```

### Recommended settings for dense/energetic tracks

```bash
python particle_visualizer.py \
  --input track.wav --output energetic.mp4 \
  --fps 60 --n_particles 2500 \
  --line_color 200 180 255 \
  --drift_speed 0.6 --damping 0.93 --damping_edge 0.85 \
  --scatter_strength 9.0 --trail_decay 0.84 \
  --fisheye_strength 0.45 --centre_brightness 0.20 \
  --spiral_rate 0.005
```

---

## All Parameters

### Input / Output

| Flag | Default | Description |
|------|---------|-------------|
| `--input` / `-i` | *(required)* | Path to input audio file (WAV recommended) |
| `--output` / `-o` | `particles.mp4` | Output video path |
| `--width` | `1920` | Output width in pixels |
| `--height` | `1080` | Output height in pixels |
| `--fps` | `60` | Output frame rate |
| `--preview` | off | Render only the first N seconds |
| `--preview_seconds` | `10.0` | Length of preview in seconds |

### Particles

| Flag | Default | Description |
|------|---------|-------------|
| `--n_particles` | `2000` | Number of particles. 1500–3000 is the sweet spot. Higher = denser field, slower render |
| `--line_color B G R` | `180 200 255` | Base particle color in BGR order. Default is cool pale blue-white |

### Motion & Physics

| Flag | Default | Description |
|------|---------|-------------|
| `--drift_speed` | `0.4` | Baseline outward speed in px/frame. Lower = slower, more viscous |
| `--damping` | `0.95` | Peak viscosity at mid-radius (0–1). Higher = more drag, slower burst decay |
| `--damping_edge` | `0.88` | Viscosity at centre and screen edge. Set equal to `--damping` for uniform drag across the field |
| `--scatter_strength` | `6.0` | Outward burst magnitude on bass hits and transients. Higher = more violent reaction |
| `--spiral_rate` | `0.003` | Rotation of drift angle in radians/frame. Positive = counter-clockwise. Negative = clockwise. `0` = straight radial lines |

### Appearance

| Flag | Default | Description |
|------|---------|-------------|
| `--trail_decay` | `0.88` | Fraction of previous frame retained (0–1). Higher = longer phosphorescent trails. `0` = no trails, each frame is clean |
| `--centre_brightness` | `0.15` | Alpha of particles at their spawn point. `0` = invisible at centre (ghostly emergence). `1` = full brightness from birth (creates a bright central blob — not recommended) |
| `--glow` / `--no_glow` | on | Enable or disable CRT phosphor glow |
| `--glow_sigma` | `7.0` | Glow blur radius in pixels. Higher = softer, more diffuse bloom |
| `--fisheye` / `--no_fisheye` | on | Enable or disable barrel lens distortion |
| `--fisheye_strength` | `0.45` | Fisheye intensity (0–1). `0` = flat. `0.7+` = strong warp |

---

## Parameter Reference: What Changes What

### Making it feel slower / more like oil
Increase `--damping` and `--damping_edge` toward `0.97–0.98`, decrease `--drift_speed` to `0.2–0.3`.

### Making the centre less bright / more mysterious
Lower `--centre_brightness` to `0.05` or `0.0`. Particles will be nearly invisible at birth and materialize as they drift outward.

### Longer, more persistent trails
Increase `--trail_decay` toward `0.93–0.95`. The field will start to feel like a long-exposure photograph.

### More dramatic reaction to bass hits
Increase `--scatter_strength` to `10–15`. Lower `--damping_edge` slightly so particles recover speed faster after the burst.

### Tighter spiral
Increase `--spiral_rate` magnitude (e.g. `--spiral_rate -0.01` for a clockwise drain effect).

### Uniform viscosity everywhere
Set `--damping_edge` equal to `--damping`. The parabola term cancels out and every particle experiences the same drag regardless of position.

---

## Output Pipeline

1. **Audio loading** — `librosa` loads the file and peak-normalises it so `scatter_strength` behaves consistently regardless of source loudness.
2. **Feature extraction** — STFT is computed once upfront. Low/mid/high band energies and onset strength are extracted per frame.
3. **Frame rendering** — OpenCV writes raw frames to a temporary silent `.mp4` via the `mp4v` codec.
4. **Audio mux** — FFmpeg re-encodes to H.264 (`libx264`, CRF 18) and muxes in the original audio as AAC 192k. The temporary file is deleted.

---

## Notes

- Input audio is **peak-normalized** before processing. This ensures `scatter_strength` and other energy-based parameters behave consistently regardless of how loud or quiet the source file is.
- The fisheye remap is computed **once per render** and applied to an output copy of each frame only — it never feeds back into the trail canvas, so distortion does not accumulate over time.
- Particle `drift_angles` are per-particle and persist across frames. The spiral rotation modifies these angles directly, so the curve is emergent from the drift steering rather than applied as a post-process.
- The trail canvas is kept in undistorted space. Glow and fisheye are applied to a copy for output only.
