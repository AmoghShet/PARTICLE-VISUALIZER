"""
Particle Field Audio Visualizer
================================
'Drowning' aesthetic — particles are born at the centre and drift outward,
fading to nothing as they reach the edges. The field is always full because
particles continuously spawn and die in a staggered cycle. It reads as an
endless, infinite dark ocean lit from somewhere behind you.

CRT phosphor glow + fisheye lens distortion.
Sister visualizer to oscilloscope_visualizer.py.

Requirements:
    pip install librosa numpy opencv-python scipy tqdm

Usage:
    # Quick 10s preview:
    python particle_visualizer.py --input Drowning.wav --output preview.mp4 --preview

    # Full render (recommended settings for Drowning):
    python particle_visualizer.py \\
        --input Drowning.wav --output drowning.mp4 \\
        --fps 60 --n_particles 2000 \\
        --line_color 180 200 255 \\
        --drift_speed 0.4 --damping 0.95 --scatter_strength 6.0 \\
        --trail_decay 0.88 --fisheye_strength 0.5
"""

import argparse
import numpy as np
import cv2
import librosa
import subprocess
import os
import tempfile
from tqdm import tqdm


# ── Fisheye ────────────────────────────────────────────────────────────────────

def build_fisheye_maps(width, height, strength=0.5):
    """Pre-compute barrel distortion remap tables (call once, reuse every frame)."""
    cx, cy = width / 2.0, height / 2.0
    xs = (np.arange(width)  - cx) / cx
    ys = (np.arange(height) - cy) / cy
    xv, yv = np.meshgrid(xs, ys)
    r     = np.sqrt(xv**2 + yv**2)
    r_src = r * (1.0 + strength * r**2)
    scale = np.where(r > 1e-8, r_src / (r + 1e-8), 1.0)
    map_x = (xv * scale * cx + cx).astype(np.float32)
    map_y = (yv * scale * cy + cy).astype(np.float32)
    return map_x, map_y


# ── Core renderer ──────────────────────────────────────────────────────────────

def render_visualizer(
    audio_path: str,
    output_path: str,
    width:  int   = 1920,
    height: int   = 1080,
    fps:    int   = 60,
    n_particles:      int   = 2000,
    line_color:       tuple = (180, 200, 255),
    glow:             bool  = True,
    glow_sigma:       float = 7.0,
    fisheye:          bool  = True,
    fisheye_strength: float = 0.45,
    drift_speed:       float = 0.4,    # outward px/frame baseline
    scatter_strength:  float = 6.0,    # extra outward burst on transient/bass hit
    damping:           float = 0.95,   # peak viscosity (at mid-radius)
    damping_edge:      float = 0.88,   # viscosity at centre and edge (faster here)
    trail_decay:       float = 0.88,   # previous-frame canvas retention
    centre_brightness: float = 0.15,   # alpha of particles at spawn (0=invisible, 1=full)
    spiral_rate:       float = 0.003,  # radians/frame rotation of drift angle
    preview_seconds:   float = None,
):
    print(f"[1/4] Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(y) / sr
    print(f"      Duration: {duration:.2f}s  |  SR: {sr} Hz")

    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak
        print(f"      Peak-normalized (original peak: {peak:.4f})")

    samples_per_frame = int(sr / fps)
    total_frames      = int(duration * fps)

    if preview_seconds is not None:
        total_frames = min(total_frames, int(preview_seconds * fps))
        print(f"      [PREVIEW] First {preview_seconds}s -> {total_frames} frames")

    # ── Per-frame audio feature extraction ───────────────────────────────────
    print("[2/4] Analysing audio ...")

    hop_length    = samples_per_frame
    n_fft         = 2048
    D             = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs         = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_stft_frames = D.shape[1]

    low_mask  = freqs < 300
    mid_mask  = (freqs >= 300) & (freqs < 3000)
    high_mask = freqs >= 3000

    def band_energy(mask):
        e  = D[mask, :].mean(axis=0)
        mx = e.max()
        return (e / mx) if mx > 1e-8 else e

    low_n  = band_energy(low_mask)
    mid_n  = band_energy(mid_mask)
    high_n = band_energy(high_mask)

    rms_frames = np.zeros(total_frames, dtype=np.float32)
    for i in range(total_frames):
        s = i * samples_per_frame
        e = min(s + samples_per_frame, len(y))
        if e > s:
            rms_frames[i] = float(np.sqrt(np.mean(y[s:e] ** 2)))
    rms_max = rms_frames.max()
    if rms_max > 1e-8:
        rms_frames /= rms_max

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_env = onset_env / (onset_env.max() + 1e-8)

    def get_frame_features(f):
        fi = min(f, n_stft_frames - 1)
        return (
            float(low_n[fi]),
            float(mid_n[fi]),
            float(high_n[fi]),
            float(rms_frames[f]) if f < len(rms_frames) else 0.0,
            float(onset_env[fi]),
        )

    # ── Geometry ──────────────────────────────────────────────────────────────
    cx_f = width  / 2.0
    cy_f = height / 2.0
    # Particles are fully invisible by the time they reach this radius.
    # Slightly beyond the corner so the fade completes before the screen edge.
    max_dist = np.sqrt(cx_f**2 + cy_f**2) * 1.05

    # ── Particle system init ──────────────────────────────────────────────────
    print(f"      Initializing {n_particles} particles ...")

    rng = np.random.default_rng(42)

    # Stagger initial positions across full radius: field is immediately full,
    # no cold-start where everything spawns from centre at once.
    init_angles = rng.uniform(0, 2 * np.pi, n_particles)
    init_dists  = rng.uniform(0, max_dist,  n_particles)
    px = (cx_f + np.cos(init_angles) * init_dists).astype(np.float64)
    py = (cy_f + np.sin(init_angles) * init_dists).astype(np.float64)

    # Each particle has a fixed personal drift angle — the direction it was born
    # on. It moves along this ray for its whole lifetime, giving the radiating
    # appearance. On respawn it gets a new random angle.
    drift_angles = init_angles.copy()
    vx = (np.cos(drift_angles) * drift_speed).astype(np.float64)
    vy = (np.sin(drift_angles) * drift_speed).astype(np.float64)

    sizes      = rng.choice([1, 1, 1, 1, 2, 2, 3], size=n_particles).astype(np.int32)
    brightness = rng.uniform(0.4, 1.0, n_particles).astype(np.float32)
    base_color = np.array(line_color, dtype=np.float32)

    if fisheye:
        print(f"      Building fisheye maps (strength={fisheye_strength}) ...")
        map_x, map_y = build_fisheye_maps(width, height, strength=fisheye_strength)

    # ── Video writer ──────────────────────────────────────────────────────────
    tmp_video = tempfile.mktemp(suffix="_noaudio.mp4")
    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    writer    = cv2.VideoWriter(tmp_video, fourcc, fps, (width, height))

    print(f"[3/4] Rendering {total_frames} frames @ {fps}fps "
          f"({'PREVIEW' if preview_seconds else 'FULL'}) ...")

    # Persistent trail canvas. Fisheye applied to a copy for output only —
    # distortion never feeds back into the trail buffer.
    trail_canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for frame_idx in tqdm(range(total_frames)):
        low, mid, high, rms, onset = get_frame_features(frame_idx)

        # ── Physics ───────────────────────────────────────────────────────────

        # Slowly rotate every particle's personal drift angle.
        # This bends the straight radial paths into gentle spirals without
        # touching velocity directly — the outward push just steers around a curve.
        if spiral_rate != 0.0:
            drift_angles -= spiral_rate   # negative = clockwise (screen y-down)

        # Compute distance from centre first — needed for variable viscosity
        dx   = px - cx_f
        dy   = py - cy_f
        dist = np.sqrt(dx**2 + dy**2)
        t    = np.clip(dist / max_dist, 0.0, 1.0).astype(np.float32)  # 0=centre, 1=edge

        # 1. Variable viscosity — damping follows a parabola peaking at t=0.5.
        #    Particles are faster near centre and edge, slowest in the middle band.
        #    4t(1-t): 0 at t=0 and t=1, peaks at 1.0 when t=0.5.
        damping_local = (damping_edge + (damping - damping_edge) * (4.0 * t * (1.0 - t))).astype(np.float64)
        vx = vx * damping_local + np.cos(drift_angles) * drift_speed * (1.0 - damping_local)
        vy = vy * damping_local + np.sin(drift_angles) * drift_speed * (1.0 - damping_local)

        # 2. Low-freq / onset: outward radial burst (cello stab shockwave)
        if onset > 0.25 or low > 0.6:
            burst = scatter_strength * max(onset, low * 0.7)
            vx += np.cos(drift_angles) * burst
            vy += np.sin(drift_angles) * burst

        # 3. Mid-freq: perpendicular swirl (slow underwater current)
        if mid > 0.4:
            swirl = mid * 0.3
            vx += -np.sin(drift_angles) * swirl
            vy +=  np.cos(drift_angles) * swirl

        # 4. High-freq: micro shimmer (string noise, reverb texture)
        if high > 0.3:
            shimmer = high * 0.25
            vx += rng.normal(0, shimmer, n_particles)
            vy += rng.normal(0, shimmer, n_particles)

        # 5. Integrate
        px += vx
        py += vy

        # Recompute dist after integration for alpha and lifecycle
        dx   = px - cx_f
        dy   = py - cy_f
        dist = np.sqrt(dx**2 + dy**2)
        t    = np.clip(dist / max_dist, 0.0, 1.0).astype(np.float32)

        # ── Visibility alpha: bell curve ──────────────────────────────────────
        # Starts at centre_brightness at spawn, rises to full brightness around
        # t≈0.35, then fades to 0 at the edge. No white blob at centre.
        rise_end    = 0.35
        rise        = np.clip(t / rise_end, 0.0, 1.0)
        rise_smooth = rise * rise * (3.0 - 2.0 * rise)   # smoothstep
        fade_out    = np.clip(1.0 - t, 0.0, 1.0)
        alpha_p     = (centre_brightness + (1.0 - centre_brightness) * rise_smooth) * fade_out

        # ── Lifecycle ─────────────────────────────────────────────────────────
        # Dead = faded out or gone offscreen → respawn at centre
        dead   = (dist >= max_dist) | (px < 0) | (px >= width) | (py < 0) | (py >= height)
        n_dead = int(dead.sum())
        if n_dead > 0:
            new_a = rng.uniform(0, 2 * np.pi, n_dead)
            drift_angles[dead] = new_a

            spawn_r = min(width, height) * 0.015
            spawn_d = rng.uniform(0, spawn_r, n_dead)
            px[dead] = cx_f + np.cos(new_a) * spawn_d
            py[dead] = cy_f + np.sin(new_a) * spawn_d

            vx[dead] = np.cos(new_a) * drift_speed
            vy[dead] = np.sin(new_a) * drift_speed

            sizes[dead]      = rng.choice([1, 1, 1, 1, 2, 2, 3], size=n_dead)
            brightness[dead] = rng.uniform(0.4, 1.0, n_dead)

        # ── Render ────────────────────────────────────────────────────────────

        # Fade trail
        np.multiply(trail_canvas, trail_decay, out=trail_canvas, casting='unsafe')

        xi = px.astype(np.int32)
        yi = py.astype(np.int32)

        # Final per-particle alpha = intrinsic brightness × distance fade
        final_alpha = brightness * alpha_p

        visible = (final_alpha > 0.02) & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        vis_idx = np.where(visible)[0]

        for i in vis_idx:
            a = float(final_alpha[i])
            color = (
                int(base_color[0] * a),
                int(base_color[1] * a),
                int(base_color[2] * a),
            )
            cv2.circle(trail_canvas, (xi[i], yi[i]), int(sizes[i]),
                       color, -1, lineType=cv2.LINE_AA)

        # CRT phosphor glow
        if glow:
            sig   = glow_sigma
            bloom = cv2.GaussianBlur(trail_canvas, (0, 0), sigmaX=sig,       sigmaY=sig)
            halo  = cv2.GaussianBlur(trail_canvas, (0, 0), sigmaX=sig * 2.5, sigmaY=sig * 2.5)
            frame_out = trail_canvas.copy()
            frame_out = cv2.add(frame_out, (halo  * 0.35).clip(0, 255).astype(np.uint8))
            frame_out = cv2.add(frame_out, (bloom * 0.60).clip(0, 255).astype(np.uint8))
        else:
            frame_out = trail_canvas.copy()

        # Fisheye — output copy only, never back into trail buffer
        if fisheye:
            frame_out = cv2.remap(frame_out, map_x, map_y,
                                  interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        writer.write(frame_out)

    writer.release()

    # ── Mux audio ─────────────────────────────────────────────────────────────
    print(f"[4/4] Muxing audio ...")

    output_duration_args = []
    if preview_seconds is not None:
        output_duration_args = ["-t", str(preview_seconds)]

    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_video,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        *output_duration_args,
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(tmp_video)

    if result.returncode != 0:
        print("[!] ffmpeg error:")
        print(result.stderr)
        raise RuntimeError("ffmpeg mux failed.")

    print(f"Done -> {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Particle field audio visualizer — 'Drowning' aesthetic"
    )
    parser.add_argument("--input",  "-i", required=True)
    parser.add_argument("--output", "-o", default="particles.mp4")
    parser.add_argument("--width",  type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps",    type=int, default=60)

    parser.add_argument("--n_particles",      type=int,   default=2000)
    parser.add_argument("--line_color",        type=int,   nargs=3, default=[180, 200, 255],
                        metavar=("B", "G", "R"))

    parser.add_argument("--drift_speed",      type=float, default=0.4,
                        help="Outward drift px/frame (default 0.4).")
    parser.add_argument("--scatter_strength", type=float, default=6.0,
                        help="Burst magnitude on bass/transient (default 6.0).")
    parser.add_argument("--damping",          type=float, default=0.95,
                        help="Peak viscosity at mid-radius 0–1 (default 0.95). Higher = slower.")
    parser.add_argument("--damping_edge",     type=float, default=0.88,
                        help="Viscosity at centre and edge 0–1 (default 0.88). "
                             "Lower than --damping = particles move faster here.")
    parser.add_argument("--trail_decay",      type=float, default=0.88,
                        help="Trail persistence 0–1 (default 0.88).")
    parser.add_argument("--centre_brightness", type=float, default=0.15,
                        help="Alpha of particles at spawn point (default 0.15). "
                             "0 = invisible at centre, 1 = full brightness from birth.")
    parser.add_argument("--spiral_rate",      type=float, default=0.003,
                        help="-Counter-clockwise rotation of drift angle in radians/frame (default 0.003). "
                             "0 = straight radial lines. 0.01 = tight spiral. Negative = clockwise.")

    parser.add_argument("--glow",             action="store_true",  default=True)
    parser.add_argument("--no_glow",          dest="glow",    action="store_false")
    parser.add_argument("--glow_sigma",       type=float, default=7.0)

    parser.add_argument("--fisheye",          action="store_true",  default=True)
    parser.add_argument("--no_fisheye",       dest="fisheye", action="store_false")
    parser.add_argument("--fisheye_strength", type=float, default=0.45)

    parser.add_argument("--preview",          action="store_true")
    parser.add_argument("--preview_seconds",  type=float, default=10.0)

    args = parser.parse_args()

    render_visualizer(
        audio_path        = args.input,
        output_path       = args.output,
        width             = args.width,
        height            = args.height,
        fps               = args.fps,
        n_particles       = args.n_particles,
        line_color        = tuple(args.line_color),
        glow              = args.glow,
        glow_sigma        = args.glow_sigma,
        fisheye           = args.fisheye,
        fisheye_strength  = args.fisheye_strength,
        drift_speed       = args.drift_speed,
        scatter_strength  = args.scatter_strength,
        damping           = args.damping,
        damping_edge      = args.damping_edge,
        trail_decay       = args.trail_decay,
        centre_brightness = args.centre_brightness,
        spiral_rate       = args.spiral_rate,
        preview_seconds   = args.preview_seconds if args.preview else None,
    )


if __name__ == "__main__":
    main()