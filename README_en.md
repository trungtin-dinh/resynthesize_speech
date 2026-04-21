<script type="text/javascript"
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$']],
      displayMath: [['$$', '$$']],
      processEscapes: true
    }
  };
</script>

# Classical Voice Analysis and Resynthesis

---

## Table of Contents

1. [The Speech Production Model](#1-the-speech-production-model)
2. [The Source-Filter Theory](#2-the-source-filter-theory)
3. [Fundamental Frequency and Voicing](#3-fundamental-frequency-and-voicing)
4. [The WORLD Vocoder](#4-the-world-vocoder)
   - 4.1 [F0 Estimation — DIO + StoneMask](#41-f0-estimation--dio--stonemask)
   - 4.2 [Spectral Envelope Estimation — CheapTrick](#42-spectral-envelope-estimation--cheaptrick)
   - 4.3 [Aperiodicity Estimation — D4C](#43-aperiodicity-estimation--d4c)
5. [Resynthesis: the Harmonic-plus-Noise Synthesis Model](#5-resynthesis-the-harmonic-plus-noise-synthesis-model)
6. [Pitch Shifting in the Vocoder Domain](#6-pitch-shifting-in-the-vocoder-domain)
7. [Time-Scale Modification via Feature Warping](#7-time-scale-modification-via-feature-warping)
8. [Short-Time Fourier Transform and Spectrogram](#8-short-time-fourier-transform-and-spectrogram)
9. [Signal Pre-processing: Resampling, Trimming, Normalisation](#9-signal-pre-processing-resampling-trimming-normalisation)
10. [Waveform Similarity via Envelope Correlation](#10-waveform-similarity-via-envelope-correlation)

---

## 1. The Speech Production Model

Human speech is the result of a physical process that can be decomposed into three distinct anatomical stages: **airflow generation**, **phonation**, and **articulation**.

The lungs provide a steady subglottal airstream. When voicing is active, the vocal folds (glottis) oscillate periodically under muscular tension and Bernoulli forces, chopping the airstream into a quasi-periodic pulse train. This pulse train is then shaped by the resonant cavities of the vocal tract — pharynx, oral cavity, nasal cavity, and lips — before radiating outward.

The vocal tract acts as an acoustic tube of time-varying shape. Its resonant frequencies, called **formants** and commonly denoted $F_1, F_2, F_3, \ldots$, are determined by the instantaneous configuration of articulators (tongue, jaw, velum, lips). These formants carry the phonetic identity of vowels and shape the spectral envelope of consonants.

This physical intuition motivates a mathematical factorisation at the heart of classical speech processing.

---

## 2. The Source-Filter Theory

The source-filter model, formalised by Fant (1960), represents the speech signal $s[n]$ as the output of a time-varying linear system driven by a source signal $e[n]$:

$$
S(z) = E(z) \cdot H(z)
$$

where $E(z)$ is the Z-transform of the excitation and $H(z)$ is the transfer function of the vocal tract filter. In the frequency domain and under the short-time stationarity assumption, this factorises the short-time power spectrum:

$$
|S(e^{j\omega})|^2 \;\approx\; |E(e^{j\omega})|^2 \cdot |H(e^{j\omega})|^2
$$

The **excitation** $e[n]$ is either:
- a **periodic pulse train** at rate $F_0$ during **voiced** phonation, or
- a **white noise process** during **unvoiced** (fricative, aspirate) segments.

The **filter** $H(e^{j\omega})$ encodes the vocal tract shape and defines the **spectral envelope** — a smooth, slowly-varying function of frequency. The spectral envelope is thus independent of $F_0$: the same vowel spoken at different pitches has the same formant structure.

This independence is the key that allows **independent manipulation** of pitch (the source periodicity) and timbre (the filter shape), which is precisely what vocoders exploit.

---

## 3. Fundamental Frequency and Voicing

The **fundamental frequency** $F_0$ (in Hz) is the inverse of the glottal period $T_0$:

$$
F_0 = \frac{1}{T_0}
$$

Typical values range from 80–180 Hz for male speakers and 160–300 Hz for female speakers. Expressed on a **musical scale**, pitch intervals are measured in **semitones**. One semitone corresponds to a frequency ratio of $2^{1/12}$, so a shift of $\Delta s$ semitones maps a frequency $F_0$ to:

$$
F_0' = F_0 \cdot 2^{\Delta s / 12}
$$

An octave up corresponds to $\Delta s = 12$, i.e. a doubling of $F_0$.

The voiced/unvoiced decision is equally important. In unvoiced frames, $F_0$ is undefined — or conventionally set to zero — and the excitation carries no harmonic structure. Any pitch manipulation must therefore act **only on voiced frames**, leaving unvoiced frames untouched to preserve the naturalness of fricatives and plosives. This selective processing is what the condition $f_0 > 0$ implements.

---

## 4. The WORLD Vocoder

WORLD (Morise et al., 2016) is a high-quality, low-latency analysis-synthesis system for speech. It decomposes the speech signal into three parametric streams, each sampled at a fixed **frame period** $T_f$ (here $T_f = 5\,\text{ms}$):

| Parameter | Symbol | Dimension per frame |
|---|---|---|
| Fundamental frequency | $F_0[m]$ | scalar (Hz or 0) |
| Spectral envelope | $\text{sp}[m, k]$ | $N_\text{fft}/2 + 1$ real values |
| Band aperiodicity | $\text{ap}[m, k]$ | $N_\text{fft}/2 + 1$ values in $[0,1]$ |

where $m$ is the frame index and $k$ is the frequency bin index.

### 4.1 F0 Estimation — DIO + StoneMask

WORLD estimates $F_0$ in two passes.

**DIO** (Distributed Inline-filter Operation) is a robust, computationally efficient $F_0$ estimator. It operates in the time domain by computing zero-crossing intervals of band-pass filtered versions of the signal. For a set of candidate $F_0$ ranges, the signal is filtered and the instantaneous period is estimated. The candidate with the highest score — measured by the degree of periodicity — is selected.

**StoneMask** is a refinement step that corrects the coarse DIO estimate. It computes the instantaneous frequency of the dominant harmonic via the phase deviation of the STFT:

$$
\hat{F}_0[m] = \arg\max_{F \in \mathcal{N}(F_0^\text{DIO}[m])} \left\{ \frac{1}{H} \sum_{h=1}^{H} \frac{|X(m, h F)|^2}{\sigma^2_\text{noise}} \right\}
$$

where $\mathcal{N}(\cdot)$ denotes a local neighbourhood, $H$ is the number of harmonics used, and $X(m, f)$ is the short-time spectrum. The result is a fine-grained, per-frame $F_0$ trajectory.

### 4.2 Spectral Envelope Estimation — CheapTrick

CheapTrick estimates the **power spectral envelope** $|H(e^{j\omega})|^2$ in a way that is robust to $F_0$ estimation errors and avoids the spectral interference from individual harmonics.

The algorithm proceeds in three steps:

**Step 1 — F0-adaptive windowing.** Around each frame, a Hanning window of length proportional to $3/F_0$ is applied. This ensures the window always spans exactly 3 fundamental periods, so the spectral resolution adapts to pitch.

**Step 2 — Power spectrum smoothing.** The squared magnitude of the windowed STFT is computed:

$$
P[m, k] = \left| \sum_{n} x[n]\, w[n - m T_f]\, e^{-j 2\pi k n / N} \right|^2
$$

A quefrency-domain liftering operation is then applied to smooth harmonic ripples. By cepstral liftering above a cutoff quefrency $q_c = 1/F_0$, the fine harmonic structure is suppressed, leaving only the smooth envelope.

**Step 3 — Spectral recovery.** The smoothed log-spectrum is constrained to be the log of a valid power spectrum by applying a correction that ensures the underlying minimum-phase filter is causal and real. This step prevents negative values under synthesis.

The output $\text{sp}[m, k]$ represents the **power** of the vocal tract envelope at frame $m$ and frequency bin $k$.

### 4.3 Aperiodicity Estimation — D4C

Natural voiced speech is never perfectly periodic. Breathiness, turbulence at the glottis, and co-articulation introduce a **stochastic component** that must be modelled explicitly.

D4C (Death, Devil, Dimension, Dilemma, and Doomsday) estimates a **band aperiodicity** $\text{ap}[m, k] \in [0, 1]$, which quantifies the ratio of noise power to total power in each frequency band at each frame. A value of $0$ means perfectly periodic (harmonic excitation only); a value of $1$ means fully aperiodic (noise excitation only).

The estimation is based on **random frequency sub-sampling**: D4C evaluates the spectral envelope at randomly offset frequency grids and measures the variance between estimates. A periodic signal has a flat harmonic structure and low inter-estimate variance; a noisy signal has high variance. The ratio of these variances is converted into the aperiodicity index.

---

## 5. Resynthesis: the Harmonic-plus-Noise Synthesis Model

Given the three parametric streams $(F_0[m], \text{sp}[m,\cdot], \text{ap}[m,\cdot])$, WORLD synthesises the output waveform using a **harmonic-plus-noise model** (HNM).

At each frame $m$, the excitation is constructed as:

$$
e[m, n] = \underbrace{\sum_{h=1}^{H[m]} A_h[m]\, \cos\!\left(2\pi h F_0[m] n / f_s + \phi_h[m]\right)}_{\text{harmonic (voiced) component}} + \underbrace{q[m, n]}_{\text{noise component}}
$$

where:
- $H[m] = \lfloor f_s / (2 F_0[m]) \rfloor$ is the number of harmonics below Nyquist,
- $A_h[m]$ is the amplitude of the $h$-th harmonic, derived from $\text{sp}[m, \cdot]$ and $\text{ap}[m, \cdot]$,
- $\phi_h[m]$ is the instantaneous phase, accumulated from frame to frame via phase integration to ensure continuity,
- $q[m, n]$ is coloured noise shaped by $\text{ap}[m, \cdot] \cdot \text{sp}[m, \cdot]$.

Specifically, the harmonic amplitude is set to:

$$
A_h[m] = \sqrt{\left(1 - \text{ap}[m, h]\right) \cdot \text{sp}[m, h]}
$$

and the noise amplitude is shaped such that its power spectral density equals $\text{ap}[m, k] \cdot \text{sp}[m, k]$.

The excitation $e[m, n]$ is then convolved with the minimum-phase filter whose power spectrum equals $\text{sp}[m, \cdot]$, and successive frames are overlap-added to produce the final time-domain signal $y[n]$.

This model guarantees that pitch, timbre, and breathiness can each be modified **independently** by altering the corresponding parameter stream before synthesis.

---

## 6. Pitch Shifting in the Vocoder Domain

Classical waveform-domain pitch shifting introduces artefacts because modifying periodicity without compensating the spectral envelope changes both pitch and timbre. In the vocoder domain, the two are **decoupled by construction**.

A pitch shift of $\Delta s$ semitones is applied by multiplying all voiced $F_0$ values by the ratio:

$$
\alpha = 2^{\Delta s / 12}
$$

Concretely, for each frame $m$ where $F_0[m] > 0$:

$$
F_0'[m] = \alpha \cdot F_0[m]
$$

The spectral envelope $\text{sp}[m, \cdot]$ and aperiodicity $\text{ap}[m, \cdot]$ are left **unchanged**. Since the formant positions are encoded in $\text{sp}$ independently of $F_0$, this manipulation raises or lowers the pitch while preserving the vowel timbre — the harmonic comb is shifted, but the envelope it rides under stays fixed.

This is in contrast to naive resampling-based pitch shifting, which modifies both $F_0$ and the formants simultaneously, producing the well-known "chipmunk effect" when speeding up or the "slow-motion" coloration when slowing down. The vocoder avoids this entirely.

---

## 7. Time-Scale Modification via Feature Warping

**Time-Scale Modification (TSM)** consists in changing the duration of a speech signal without altering its pitch or spectral content. In the vocoder domain, this is achieved by **temporal interpolation** of the parameter frames.

Let $M$ be the original number of frames and $M'$ the target number of frames, related by a speed factor $\lambda > 0$ such that:

$$
M' = \left\lfloor \frac{M}{\lambda} \right\rfloor
$$

A speed factor $\lambda > 1$ compresses the signal in time (faster speech); $\lambda < 1$ stretches it (slower speech).

Each parameter stream is resampled from $M$ to $M'$ frames by **linear interpolation** on a normalized time grid. For a 1-D stream $\theta[m]$ (such as $F_0$), the resampled value at new frame $m'$ is:

$$
\theta'[m'] = (1 - \alpha)\,\theta[\lfloor m \rfloor] + \alpha\,\theta[\lceil m \rceil]
\quad \text{where} \quad m = m' \cdot \frac{M-1}{M'-1}, \quad \alpha = m - \lfloor m \rfloor
$$

For the 2-D streams $\text{sp}[m, k]$ and $\text{ap}[m, k]$, the same interpolation is applied independently along the time axis for each frequency bin $k$.

This is a simple but effective approach because the WORLD parameter streams are **slowly varying** relative to the frame period of 5 ms. Linear interpolation introduces negligible artefacts for moderate speed factors ($0.5 \leq \lambda \leq 2.0$). More aggressive warping would benefit from cubic or sinc interpolation, but linear suffices for the perceptual quality targeted here.

Note that the synthesised output duration is approximately $M' \cdot T_f$ seconds, and the fundamental frequency trajectory is preserved at each new frame. Crucially, the spectral envelope is also resampled (not stretched), so the formant structure remains physically consistent throughout the modified utterance.

---

## 8. Short-Time Fourier Transform and Spectrogram

The **Short-Time Fourier Transform (STFT)** of a discrete-time signal $x[n]$ is defined as:

$$
X[m, k] = \sum_{n=-\infty}^{+\infty} x[n]\, w[n - m H]\, e^{-j 2\pi k n / N}
$$

where $w[\cdot]$ is an analysis window (here a Hann window), $H$ is the hop size in samples, $N$ is the FFT size (here $N = 1024$), $m$ is the frame index, and $k \in \{0, 1, \ldots, N/2\}$ is the frequency bin.

The **spectrogram** is the squared magnitude of the STFT, converted to decibels:

$$
P_\text{dB}[m, k] = 20 \log_{10}\!\left(\max\!\left(|X[m, k]|,\, \epsilon\right)\right)
$$

where $\epsilon = 10^{-8}$ is a small constant preventing $\log(0)$. The factor $20$ (rather than $10$) is used because $|X[m,k]|$ is an amplitude spectrum rather than a power spectrum.

The **frequency resolution** of the STFT is $\Delta f = f_s / N$, and the **time resolution** per frame is $\Delta t = H / f_s$. There is a fundamental time-frequency trade-off: increasing $N$ improves frequency resolution but reduces temporal resolution and vice versa.

In the spectrogram display, the $F_0$ contour extracted by WORLD is overlaid as a cyan curve, serving as a ground-truth reference for the dominant harmonic. Visually, the $F_0$ contour should track the first harmonic visible in the spectrogram — deviations signal either estimation error or non-stationarity.

---

## 9. Signal Pre-processing: Resampling, Trimming, Normalisation

Before analysis, several pre-processing steps are applied to guarantee numerical consistency.

**Resampling.** Audio signals from different sources may carry different sampling rates $f_s^\text{orig}$. All signals are resampled to a common target rate $f_s$ (default: 24 000 Hz) using a polyphase anti-aliasing filter (as implemented in `librosa.resample`). Resampling applies a low-pass filter at $f_s/2$ to prevent aliasing, then interpolates or decimates. The resampling ratio is $r = f_s / f_s^\text{orig}$, and the filter length scales with $1/\min(r, 1)$ to maintain a flat passband up to the lower Nyquist frequency.

**Silence trimming.** Leading and trailing silence is removed by thresholding the short-time RMS energy. Frames with energy below $-T_\text{dB} = -25\,\text{dB}$ relative to the peak are discarded. This prevents WORLD from estimating $F_0$ over silent regions, which would introduce spurious voiced frames.

**DC removal.** The sample mean $\bar{x} = \frac{1}{N}\sum_n x[n]$ is subtracted from the signal. DC offset arises from microphone bias or DC-coupled amplifiers. It has no perceptual content and would bias the short-time energy estimator.

**Peak normalisation.** The signal is scaled to a peak amplitude of 0.98:

$$
x_\text{norm}[n] = 0.98 \cdot \frac{x[n]}{\max_n |x[n]|}
$$

This ensures a consistent input dynamic range for WORLD and prevents numerical overflow in single-precision synthesis.

---

## 10. Waveform Similarity via Envelope Correlation

To provide a perceptually meaningful comparison between the original signal $x[n]$ and the resynthesised signal $y[n]$, similarity is measured not at the waveform level but at the **amplitude envelope** level.

**Motivation.** The resynthesised signal is produced with independently randomised phase noise in the aperiodic component. Waveform-level cross-correlation would therefore give low values even for a perfect resynthesis, since random phase shifts decorrelate the fine structure while leaving perception unchanged. The amplitude envelope, by contrast, tracks the slow energy modulation of speech — syllabic rhythm, prosodic contour, pause structure — and is largely invariant to fine-structure phase.

**Envelope estimation.** The amplitude envelope $\hat{e}[n]$ of a signal $x[n]$ is estimated by full-wave rectification followed by moving-average smoothing with a rectangular window of length $L$ (default: $L = f_s \cdot 20\,\text{ms}$):

$$
\hat{e}[n] = \frac{1}{L} \sum_{\ell=0}^{L-1} |x[n - \ell]|
$$

This is equivalent to low-pass filtering the rectified signal with a brick-wall filter of cutoff $f_c = f_s / L \approx 50\,\text{Hz}$, which retains syllabic amplitude modulation (typically 4–8 Hz for speech) while suppressing fine pitch-level fluctuations.

**Alignment.** The resynthesised signal may have a slightly different duration due to synthesis frame overlap-add. It is aligned to the reference length by linear interpolation of the full envelope vector onto the same normalised time grid used for TSM.

**Normalised correlation.** The similarity metric is the **Pearson correlation coefficient** of the mean-subtracted envelopes:

$$
\rho = \frac{(\hat{e}_x - \bar{\hat{e}}_x)^\top (\hat{e}_y - \bar{\hat{e}}_y)}{\|\hat{e}_x - \bar{\hat{e}}_x\|_2 \cdot \|\hat{e}_y - \bar{\hat{e}}_y\|_2}
$$

This is clipped to $[0, 1]$ and reported as a percentage. A value near 100% indicates that the temporal energy contour of the resynthesised signal closely matches the original — the vocoder has faithfully reproduced the speaking rhythm and prosodic dynamics. Values below ~70% typically indicate significant artefacts or a failure of the analysis-synthesis cycle.

---

## References

- Fant, G. (1960). *Acoustic Theory of Speech Production*. Mouton.
- Morise, M., Yokomori, F., & Ozawa, K. (2016). WORLD: a vocoder-based high-quality speech synthesis system for real-time applications. *IEICE Transactions on Information and Systems*, E99-D(7), 1877–1884.
- Kawahara, H. (2006). STRAIGHT, exploitation of the other aspect of VOCODER. *Phonetical Sciences*, 12, 21–36.
- McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis based on a sinusoidal representation. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 34(4), 744–754.
- Driedger, J., & Müller, M. (2016). A review of time-scale modification of music signals. *Applied Sciences*, 6(2), 57.
