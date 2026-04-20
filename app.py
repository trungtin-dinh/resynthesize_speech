import gradio as gr
import librosa
import numpy as np
import plotly.graph_objects as go
import pyworld as pw

DEFAULT_TARGET_SR = 24000
MIN_TARGET_SR = 16000
MAX_TARGET_SR = 96000
FRAME_PERIOD_MS = 5.0
TRIM_TOP_DB = 25
SPECTROGRAM_NFFT = 1024
ENVELOPE_WINDOW_MS = 20.0
NORMALIZATION_PEAK = 0.98

REFERENCE_SENTENCE = (
    "Today we measure how a voice changes in pitch, resonance, and timing during natural speech."
)

LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]

def sanitize_target_sr(target_sr):
    if target_sr is None:
        target_sr = DEFAULT_TARGET_SR

    try:
        target_sr = int(round(float(target_sr)))
    except Exception as exc:
        raise gr.Error("The target sample rate must be a valid integer.") from exc

    if target_sr < MIN_TARGET_SR:
        raise gr.Error(
            f"Please use a target sample rate greater than or equal to {MIN_TARGET_SR} Hz."
        )
    if target_sr > MAX_TARGET_SR:
        raise gr.Error(
            f"Please use a target sample rate lower than or equal to {MAX_TARGET_SR} Hz."
        )

    return target_sr


def sanitize_pitch_shift(pitch_shift_semitones):
    if pitch_shift_semitones is None:
        return 0.0

    try:
        return float(pitch_shift_semitones)
    except Exception as exc:
        raise gr.Error("Pitch shift must be a valid number.") from exc


def sanitize_speed_factor(speed_factor):
    if speed_factor is None:
        return 1.0

    try:
        speed_factor = float(speed_factor)
    except Exception as exc:
        raise gr.Error("Speed factor must be a valid number.") from exc

    if speed_factor <= 0:
        raise gr.Error("Speed factor must be strictly positive.")

    return speed_factor


def load_audio(audio_path: str, target_sr: int):
    if audio_path is None:
        raise gr.Error("Please record or upload a voice sample first.")

    x, sr = librosa.load(audio_path, sr=None, mono=True)
    if x.size == 0:
        raise gr.Error("The audio file is empty.")

    x, _ = librosa.effects.trim(x, top_db=TRIM_TOP_DB)
    if x.size == 0:
        raise gr.Error("The recording is too silent. Please speak a little louder.")

    if sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    x = x.astype(np.float64)
    x = x - np.mean(x)

    peak = np.max(np.abs(x))
    if peak > 0:
        x = NORMALIZATION_PEAK * x / peak

    return x, sr


def interp_rows(matrix: np.ndarray, new_len: int) -> np.ndarray:
    old_len = matrix.shape[0]
    if old_len == new_len:
        return matrix.copy()

    if old_len < 2:
        return np.repeat(matrix, new_len, axis=0)

    old_grid = np.linspace(0.0, 1.0, old_len)
    new_grid = np.linspace(0.0, 1.0, new_len)

    out = np.empty((new_len, matrix.shape[1]), dtype=np.float64)
    for j in range(matrix.shape[1]):
        out[:, j] = np.interp(new_grid, old_grid, matrix[:, j])

    return out


def warp_features(f0, sp, ap, speed_factor: float):
    n_frames = len(f0)
    new_frames = max(2, int(round(n_frames / speed_factor)))

    old_grid = np.linspace(0.0, 1.0, n_frames)
    new_grid = np.linspace(0.0, 1.0, new_frames)

    new_f0 = np.interp(new_grid, old_grid, f0)
    new_sp = interp_rows(sp, new_frames)
    new_ap = interp_rows(ap, new_frames)

    return new_f0, new_sp, new_ap


def align_signal_to_reference(reference: np.ndarray, signal: np.ndarray) -> np.ndarray:
    if len(signal) == len(reference):
        return signal.astype(np.float64)

    if len(signal) == 0:
        return np.zeros_like(reference, dtype=np.float64)

    if len(signal) == 1:
        return np.full_like(reference, float(signal[0]), dtype=np.float64)

    old_grid = np.linspace(0.0, 1.0, len(signal))
    new_grid = np.linspace(0.0, 1.0, len(reference))

    return np.interp(new_grid, old_grid, signal).astype(np.float64)


def smooth_envelope(signal: np.ndarray, sr: int, window_ms: float = ENVELOPE_WINDOW_MS) -> np.ndarray:
    signal = np.abs(signal.astype(np.float64))
    win_len = max(3, int(sr * window_ms / 1000.0))
    if win_len % 2 == 0:
        win_len += 1

    kernel = np.ones(win_len, dtype=np.float64) / win_len
    return np.convolve(signal, kernel, mode="same")


def compute_waveform_similarity_percent(x: np.ndarray, y: np.ndarray, sr: int) -> float:
    y_aligned = align_signal_to_reference(x, y)

    env_x = smooth_envelope(x, sr)
    env_y = smooth_envelope(y_aligned, sr)

    env_x = env_x - np.mean(env_x)
    env_y = env_y - np.mean(env_y)

    denom = np.linalg.norm(env_x) * np.linalg.norm(env_y)
    if denom <= 1e-12:
        return 0.0

    similarity = float(np.dot(env_x, env_y) / denom)
    similarity = float(np.clip(similarity, 0.0, 1.0))

    return 100.0 * similarity


def compute_mean_f0(f0: np.ndarray):
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        return None
    return float(np.mean(voiced))


def make_waveform_figure(x, sr, y):
    tx = np.arange(len(x)) / sr
    ty = np.arange(len(y)) / sr
    similarity_percent = compute_waveform_similarity_percent(x, y, sr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tx, y=x, mode="lines", name="Original"))
    fig.add_trace(go.Scatter(x=ty, y=y, mode="lines", name="Resynthesized"))

    fig.update_layout(
        title=f"Time-domain waveforms<br><sup>Similarity (envelope match) = {similarity_percent:.1f}%</sup>",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        legend_title="Signal",
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


def make_spectrogram_figure(x, sr, f0):
    hop_length = int(sr * FRAME_PERIOD_MS / 1000.0)
    if hop_length <= 0:
        hop_length = 120

    stft = librosa.stft(x.astype(np.float32), n_fft=SPECTROGRAM_NFFT, hop_length=hop_length)
    mag = np.abs(stft)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-8))

    freqs = np.linspace(0, sr / 2, mag_db.shape[0])
    times = np.arange(mag_db.shape[1]) * hop_length / sr

    mean_f0 = compute_mean_f0(f0)
    if mean_f0 is None:
        subtitle = "Estimated mean F0 = unavailable"
    else:
        subtitle = f"Estimated mean F0 = {mean_f0:.1f} Hz"

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=mag_db,
            x=times,
            y=freqs,
            colorbar_title="dB",
        )
    )

    f0_times = np.arange(len(f0)) * FRAME_PERIOD_MS / 1000.0
    f0_plot = np.where(f0 > 0, f0, np.nan)

    fig.add_trace(
        go.Scatter(
            x=f0_times,
            y=f0_plot,
            mode="lines",
            name="F0",
            line=dict(color="cyan", width=2),
        )
    )

    fig.update_layout(
        title=f"Spectrogram<br><sup>{subtitle}</sup>",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


def analyze_and_resynthesize(audio_path, pitch_shift_semitones, speed_factor, target_sr):
    pitch_shift_semitones = sanitize_pitch_shift(pitch_shift_semitones)
    speed_factor = sanitize_speed_factor(speed_factor)
    target_sr = sanitize_target_sr(target_sr)

    x, sr = load_audio(audio_path, target_sr)

    f0, sp, ap = pw.wav2world(
        np.ascontiguousarray(x, dtype=np.float64),
        sr,
        frame_period=FRAME_PERIOD_MS,
    )

    pitch_ratio = 2.0 ** (pitch_shift_semitones / 12.0)

    voiced = f0 > 0
    f0_mod = f0.copy()
    f0_mod[voiced] *= pitch_ratio

    f0_mod, sp_mod, ap_mod = warp_features(f0_mod, sp, ap, speed_factor)

    y = pw.synthesize(f0_mod, sp_mod, ap_mod, sr, frame_period=FRAME_PERIOD_MS)
    y = np.asarray(y, dtype=np.float32)

    peak = np.max(np.abs(y))
    if peak > 0:
        y = NORMALIZATION_PEAK * y / peak

    spectrogram_fig = make_spectrogram_figure(x, sr, f0)
    waveform_fig = make_waveform_figure(x, sr, y)

    return (sr, y), spectrogram_fig, waveform_fig


with gr.Blocks(title="Classical Voice Analysis and Resynthesis") as demo:
    gr.Markdown(
        f"""
### Reference sentence
> Example: {REFERENCE_SENTENCE}
"""
    )

    with gr.Row():
        with gr.Column(scale=3):
            audio_in = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Voice input",
            )

        with gr.Column(scale=2):
            pitch_shift = gr.Number(
                value=0.0,
                precision=1,
                step=0.5,
                label="Pitch shift (semitones)",
            )
            speed_factor = gr.Number(
                value=1.0,
                precision=1,
                minimum=0.1,
                step=0.1,
                label="Speed factor",
            )
            target_sr_input = gr.Number(
                value=DEFAULT_TARGET_SR,
                precision=0,
                minimum=MIN_TARGET_SR,
                maximum=MAX_TARGET_SR,
                step=1000,
                label="Target sample rate (Hz)",
            )
            analyze_button = gr.Button("Analyze and resynthesize")

        with gr.Column(scale=3):
            audio_out = gr.Audio(label="Resynthesized voice")

    with gr.Row():
        with gr.Column(scale=3):
            spectrogram_plot = gr.Plot(label="Spectrogram")
        with gr.Column(scale=3):
            waveform_plot = gr.Plot(label="Time plot")

    analyze_button.click(
        fn=analyze_and_resynthesize,
        inputs=[audio_in, pitch_shift, speed_factor, target_sr_input],
        outputs=[audio_out, spectrogram_plot, waveform_plot],
    )

if __name__ == "__main__":
    demo.launch()