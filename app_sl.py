import io
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import plotly.graph_objects as go
import pyworld as pw
import soundfile as sf
import streamlit as st

DEFAULT_TARGET_SR = 24000
MIN_TARGET_SR = 16000
MAX_TARGET_SR = 48000
FRAME_PERIOD_MS = 5.0
TRIM_TOP_DB = 25
SPECTROGRAM_NFFT = 1024
ENVELOPE_WINDOW_MS = 20.0
NORMALIZATION_PEAK = 0.98
DEFAULT_AUDIO_URL = "https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"


class AppError(Exception):
    pass


def read_text_file(path: str, fallback: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return fallback
    return file_path.read_text(encoding="utf-8")


DOCUMENTATION_fr = read_text_file(
    "documentation_fr.md",
    "## Documentation FR\n\nThe file `documentation_fr.md` was not found next to `app_sl.py`.",
)

DOCUMENTATION_en = read_text_file(
    "documentation_en.md",
    "## Documentation EN\n\nThe file `documentation_en.md` was not found next to `app_sl.py`.",
)


def split_markdown_by_h2(markdown_text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    parts = re.split(r"(?m)^##\s+", markdown_text.strip())

    for part in parts:
        part = part.strip()
        if not part:
            continue

        lines = part.splitlines()
        title = lines[0].strip()

        if title.lower() in {"table des matières", "table of contents"}:
            continue

        sections[title] = "## " + part

    if not sections and markdown_text.strip():
        sections["Documentation"] = markdown_text.strip()

    return sections


DOC_FR_SECTIONS = split_markdown_by_h2(DOCUMENTATION_fr)
DOC_EN_SECTIONS = split_markdown_by_h2(DOCUMENTATION_en)

DOC_FR_TITLES = list(DOC_FR_SECTIONS.keys())
DOC_EN_TITLES = list(DOC_EN_SECTIONS.keys())


def sanitize_target_sr(target_sr):
    if target_sr is None:
        target_sr = DEFAULT_TARGET_SR

    try:
        target_sr = int(round(float(target_sr)))
    except Exception as exc:
        raise AppError("The target sample rate must be a valid integer.") from exc

    if target_sr < MIN_TARGET_SR:
        raise AppError(
            f"Please use a target sample rate greater than or equal to {MIN_TARGET_SR} Hz."
        )
    if target_sr > MAX_TARGET_SR:
        raise AppError(
            f"Please use a target sample rate lower than or equal to {MAX_TARGET_SR} Hz."
        )

    return target_sr


def sanitize_pitch_shift(pitch_shift_semitones):
    if pitch_shift_semitones is None:
        return 0.0

    try:
        return float(pitch_shift_semitones)
    except Exception as exc:
        raise AppError("Pitch shift must be a valid number.") from exc


def sanitize_speed_factor(speed_factor):
    if speed_factor is None:
        return 1.0

    try:
        speed_factor = float(speed_factor)
    except Exception as exc:
        raise AppError("Speed factor must be a valid number.") from exc

    if speed_factor <= 0:
        raise AppError("Speed factor must be strictly positive.")

    return speed_factor


def load_audio(audio_path: str, target_sr: int):
    if audio_path is None:
        raise AppError("Please record or upload a voice sample first.")

    x, sr = librosa.load(audio_path, sr=None, mono=True)
    if x.size == 0:
        raise AppError("The audio file is empty.")

    x, _ = librosa.effects.trim(x, top_db=TRIM_TOP_DB)
    if x.size == 0:
        raise AppError("The recording is too silent. Please speak a little louder.")

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
        return np.tile(matrix[:1], (new_len, 1))

    old_grid = np.linspace(0.0, 1.0, old_len)
    new_grid = np.linspace(0.0, 1.0, new_len)

    idx = np.searchsorted(old_grid, new_grid, side="right") - 1
    idx = np.clip(idx, 0, old_len - 2)

    alpha = (new_grid - old_grid[idx]) / (old_grid[idx + 1] - old_grid[idx])

    return (
        (1.0 - alpha[:, None]) * matrix[idx] + alpha[:, None] * matrix[idx + 1]
    ).astype(np.float64)


def warp_features(f0: np.ndarray, sp: np.ndarray, ap: np.ndarray, speed_factor: float):
    n_frames = len(f0)
    new_frames = max(2, int(round(n_frames / speed_factor)))

    old_grid = np.linspace(0.0, 1.0, n_frames)
    new_grid = np.linspace(0.0, 1.0, new_frames)

    voiced_mask = (f0 > 0).astype(np.float64)
    new_voiced = np.interp(new_grid, old_grid, voiced_mask) >= 0.5

    new_f0 = np.interp(new_grid, old_grid, f0)
    new_f0[~new_voiced] = 0.0

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
        title=f"Time-domain waveforms<br><sup>Envelope similarity = {similarity_percent:.1f}%</sup>",
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
    subtitle = "Estimated mean F0 = unavailable" if mean_f0 is None else f"Estimated mean F0 = {mean_f0:.1f} Hz"

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
        title=f"Spectrogram with F0 contour<br><sup>{subtitle}</sup>",
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

    return sr, y, spectrogram_fig, waveform_fig


def save_uploaded_file(uploaded_file, suffix: Optional[str] = None) -> str:
    if uploaded_file is None:
        raise AppError("Please upload or record a voice sample first.")

    file_suffix = suffix or Path(uploaded_file.name or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


@st.cache_data(show_spinner=False)
def download_default_audio() -> bytes:
    with urllib.request.urlopen(DEFAULT_AUDIO_URL, timeout=20) as response:
        return response.read()


def write_default_audio_to_temp() -> str:
    try:
        audio_bytes = download_default_audio()
    except Exception as exc:
        raise AppError(
            "Default audio could not be loaded. Please upload or record a local audio file instead."
        ) from exc

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        return tmp.name


def audio_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()


def safe_key_fragment(text: str) -> str:
    key = re.sub(r"[^0-9A-Za-z_]+", "_", text.strip())
    key = key.strip("_")
    return key[:48] or "section"


def set_doc_section(selected_key: str, title: str):
    st.session_state[selected_key] = title


def render_documentation(sections: dict[str, str], titles: list[str], key_prefix: str):
    if not titles:
        st.warning("No documentation section was found.")
        return

    selected_key = f"{key_prefix}_selected_doc_title"
    if selected_key not in st.session_state or st.session_state[selected_key] not in sections:
        st.session_state[selected_key] = titles[0]

    nav_col, text_col = st.columns([1, 2])

    with nav_col:
        for idx, title in enumerate(titles):
            is_selected = title == st.session_state[selected_key]
            st.button(
                title,
                key=f"{key_prefix}_doc_button_{idx}_{safe_key_fragment(title)}",
                type="primary" if is_selected else "secondary",
                use_container_width=True,
                on_click=set_doc_section,
                args=(selected_key, title),
            )

    with text_col:
        selected_title = st.session_state[selected_key]
        st.markdown(sections[selected_title])


def render_analysis_tab():
    input_col, control_col, output_col = st.columns([3, 2, 3])

    with input_col:
        st.markdown("### Voice input")
        input_modes = ["Default sample", "Upload audio"]
        if hasattr(st, "audio_input"):
            input_modes.append("Record audio")

        input_mode = st.radio(
            "Input source",
            input_modes,
            horizontal=True,
            key="input_mode",
        )

        uploaded_audio = None
        recorded_audio = None

        if input_mode == "Default sample":
            st.audio(DEFAULT_AUDIO_URL)
        elif input_mode == "Upload audio":
            uploaded_audio = st.file_uploader(
                "Upload a voice sample",
                type=["wav", "mp3", "ogg", "flac", "m4a", "aac"],
                key="uploaded_audio",
            )
            if uploaded_audio is not None:
                st.audio(uploaded_audio)
        else:
            recorded_audio = st.audio_input("Record a voice sample", key="recorded_audio")
            if recorded_audio is not None:
                st.audio(recorded_audio)

    with control_col:
        st.markdown("### Parameters")
        pitch_shift = st.number_input(
            "Pitch shift (semitones)",
            value=0.0,
            step=0.5,
            format="%.1f",
        )
        speed_factor = st.number_input(
            "Speed factor",
            min_value=0.1,
            value=1.0,
            step=0.1,
            format="%.1f",
        )
        target_sr = st.number_input(
            "Target sample rate (Hz)",
            min_value=MIN_TARGET_SR,
            max_value=MAX_TARGET_SR,
            value=DEFAULT_TARGET_SR,
            step=1000,
            format="%d",
        )
        run_button = st.button("Analyze and resynthesize", type="primary", use_container_width=True)

    if run_button:
        audio_path = None
        try:
            if input_mode == "Default sample":
                audio_path = write_default_audio_to_temp()
            elif input_mode == "Upload audio":
                audio_path = save_uploaded_file(uploaded_audio)
            else:
                audio_path = save_uploaded_file(recorded_audio, suffix=".wav")

            with st.spinner("Analyzing and resynthesizing..."):
                sr, y, spectrogram_fig, waveform_fig = analyze_and_resynthesize(
                    audio_path,
                    pitch_shift,
                    speed_factor,
                    target_sr,
                )

            st.session_state["result_sr"] = sr
            st.session_state["result_audio"] = y
            st.session_state["spectrogram_fig"] = spectrogram_fig
            st.session_state["waveform_fig"] = waveform_fig
            st.session_state["result_wav_bytes"] = audio_to_wav_bytes(y, sr)

        except AppError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

    with output_col:
        st.markdown("### Resynthesized voice")
        if "result_wav_bytes" in st.session_state and "result_sr" in st.session_state:
            st.audio(
                st.session_state["result_wav_bytes"],
                format="audio/wav",
            )
            st.download_button(
                "Download resynthesized WAV",
                data=st.session_state["result_wav_bytes"],
                file_name="resynthesized_voice.wav",
                mime="audio/wav",
                use_container_width=True,
            )
        else:
            st.info("Run the analysis to display the resynthesized voice.")

    plot_col_1, plot_col_2 = st.columns([3, 3])
    with plot_col_1:
        if "spectrogram_fig" in st.session_state:
            st.plotly_chart(st.session_state["spectrogram_fig"], use_container_width=True)
        else:
            st.info("The spectrogram with F0 contour will appear here.")

    with plot_col_2:
        if "waveform_fig" in st.session_state:
            st.plotly_chart(st.session_state["waveform_fig"], use_container_width=True)
        else:
            st.info("The time-domain waveforms will appear here.")


def main():
    st.set_page_config(
        page_title="Classical Voice Analysis and Resynthesis",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 5.2rem !important;
            padding-bottom: 2rem !important;
        }
        header[data-testid="stHeader"] {
            background: #0e1117;
        }
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stPlotlyChart"]) {
            border-radius: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    analysis_tab, doc_fr_tab, doc_en_tab = st.tabs(
        ["Analysis & Resynthesis", "Documentation FR", "Documentation EN"]
    )

    with analysis_tab:
        render_analysis_tab()

    with doc_fr_tab:
        render_documentation(DOC_FR_SECTIONS, DOC_FR_TITLES, "fr")

    with doc_en_tab:
        render_documentation(DOC_EN_SECTIONS, DOC_EN_TITLES, "en")


if __name__ == "__main__":
    main()
