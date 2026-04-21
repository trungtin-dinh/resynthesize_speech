## Table of Contents

1. [The Speech Production Model](#1-the-speech-production-model)
2. [The Source-Filter Theory](#2-the-source-filter-theory)
3. [Fundamental Frequency and Voicing](#3-fundamental-frequency-and-voicing)
4. [The WORLD Vocoder](#4-the-world-vocoder)
5. [Resynthesis: the Harmonic-plus-Noise Model](#5-resynthesis-the-harmonic-plus-noise-synthesis-model)
6. [Pitch Shifting in the Vocoder Domain](#6-pitch-shifting-in-the-vocoder-domain)
7. [Time-Scale Modification via Feature Warping](#7-time-scale-modification-via-feature-warping)
8. [Short-Time Fourier Transform and Spectrogram](#8-short-time-fourier-transform-and-spectrogram)
9. [Signal Pre-processing](#9-signal-pre-processing-resampling-trimming-normalisation)
10. [Waveform Similarity via Envelope Correlation](#10-waveform-similarity-via-envelope-correlation)

---

## 1. The Speech Production Model

Human speech results from a physical process decomposable into three distinct anatomical stages:
**airflow generation**, **phonation**, and **articulation**.

The lungs provide a steady subglottal airstream. When voicing is active, the vocal folds (glottis)
oscillate periodically under muscular tension and Bernoulli forces, chopping the airstream into a
quasi-periodic pulse train. This pulse train is then shaped by the resonant cavities of the vocal
tract — pharynx, oral cavity, nasal cavity, and lips — before radiating outward.

The vocal tract acts as an acoustic tube of time-varying shape. Its resonant frequencies, called
**formants** and commonly denoted $F_1, F_2, F_3, \ldots$, are determined by the instantaneous
configuration of articulators (tongue, jaw, velum, lips). These formants carry the phonetic identity
of vowels and shape the spectral envelope of consonants.

This physical intuition motivates a mathematical factorisation at the heart of classical speech
processing.

---

## 2. The Source-Filter Theory

The source-filter model, formalised by Fant (1960), represents the speech signal $s[n]$ as the
output of a time-varying linear system driven by a source signal $e[n]$:

$$
S(z) = E(z) \cdot H(z)
$$

where $E(z)$ is the Z-transform of the excitation and $H(z)$ is the transfer function of the vocal
tract filter. In the frequency domain and under the short-time stationarity assumption, this
factorises the short-time power spectrum:

$$
|S(e^{j\omega})|^2 \;\approx\; |E(e^{j\omega})|^2 \cdot |H(e^{j\omega})|^2
$$

The **excitation** $e[n]$ is either:
- a **periodic pulse train** at rate $F_0$ during **voiced** phonation, or
- a **white noise process** during **unvoiced** (fricative, aspirate) segments.

The **filter** $H(e^{j\omega})$ encodes the vocal tract shape and defines the **spectral envelope** —
a smooth, slowly-varying function of frequency. Crucially, the spectral envelope is independent of
$F_0$: the same vowel spoken at different pitches has the same formant structure.

This independence is the key that allows **independent manipulation** of pitch (source periodicity)
and timbre (filter shape), which is precisely what vocoders exploit.

---

## 3. Fundamental Frequency and Voicing

The **fundamental frequency** $F_0$ (in Hz) is the inverse of the glottal period $T_0$:

$$
F_0 = \frac{1}{T_0}
$$

Typical values range from 80-180 Hz for male speakers and 160-300 Hz for female speakers. Expressed
on a **musical scale**, pitch intervals are measured in **semitones**. One semitone corresponds to a
frequency ratio of $2^{1/12}$, so a shift of $\Delta s$ semitones maps a frequency $F_0$ to:

$$
F_0' = F_0 \cdot 2^{\Delta s / 12}
$$

An octave up corresponds to $\Delta s = 12$, i.e. a doubling of $F_0$.

The voiced/unvoiced decision is equally important. In unvoiced frames, $F_0$ is undefined — or
conventionally set to zero — and the excitation carries no harmonic structure. Any pitch
manipulation must therefore act **only on voiced frames** (where $F_0 > 0$), leaving unvoiced
frames untouched to preserve the naturalness of fricatives and plosives.

---

## 4. The WORLD Vocoder

WORLD (Morise et al., 2016) is a high-quality, low-latency analysis-synthesis system for speech.
It decomposes the speech signal into three parametric streams, each sampled at a fixed
**frame period** $T_f$ (here $T_f = 5\,\text{ms}$):

| Parameter | Symbol | Dimension per frame |
|---|---|---|
| Fundamental frequency | $F_0[m]$ | scalar (Hz, or 0 if unvoiced) |
| Spectral envelope | $\text{sp}[m, k]$ | $N_\text{fft}/2 + 1$ real values |
| Band aperiodicity | $\text{ap}[m, k]$ | $N_\text{fft}/2 + 1$ values in $[0,1]$ |

where $m$ is the frame index and $k$ is the frequency bin index.

### 4.1 F0 Estimation — DIO + StoneMask

WORLD estimates $F_0$ in two passes.

**DIO** (Distributed Inline-filter Operation) is a robust, computationally efficient $F_0$
estimator. It operates in the time domain by computing zero-crossing intervals of band-pass filtered
versions of the signal. For a set of candidate $F_0$ ranges, the signal is filtered and the
instantaneous period is estimated. The candidate with the highest periodicity score is selected.

**StoneMask** is a refinement step that corrects the coarse DIO estimate via instantaneous frequency
estimation on the dominant harmonic. For each frame $m$, the refined estimate is:

$$
\hat{F}_0[m] = \arg\max_{F \in \mathcal{N}(F_0^{\text{DIO}}[m])}
\left\{ \frac{1}{H} \sum_{h=1}^{H} \frac{|X(m,\, hF)|^2}{\sigma^2_{\text{noise}}} \right\}
$$

where $\mathcal{N}(\cdot)$ denotes a local neighbourhood around the DIO estimate, $H$ is the number
of harmonics used, and $X(m, f)$ is the short-time spectrum at frame $m$ and frequency $f$.

### 4.2 Spectral Envelope Estimation — CheapTrick

CheapTrick estimates the **power spectral envelope** $|H(e^{j\omega})|^2$ in a way that is robust
to $F_0$ estimation errors and avoids spectral interference from individual harmonics.

**Step 1 — F0-adaptive windowing.** A Hanning window of length proportional to $3/F_0$ is applied
around each frame, ensuring the window always spans exactly 3 fundamental periods so spectral
resolution adapts to pitch.

**Step 2 — Power spectrum smoothing.** The squared magnitude of the windowed STFT is computed:

$$
P[m, k] = \left| \sum_{n} x[n]\, w[n - m T_f]\, e^{-j 2\pi k n / N} \right|^2
$$

A quefrency-domain liftering operation then suppresses harmonic ripples. By cepstral liftering above
cutoff quefrency $q_c = 1/F_0$, the fine harmonic structure is removed, leaving only the smooth
envelope.

**Step 3 — Spectral recovery.** The smoothed log-spectrum is constrained to be the log of a valid
power spectrum via a minimum-phase correction. This step prevents negative spectral values under
synthesis.

### 4.3 Aperiodicity Estimation — D4C

Natural voiced speech is never perfectly periodic. Breathiness, glottal turbulence, and
co-articulation introduce a **stochastic component** modelled by the **band aperiodicity**
$\text{ap}[m, k] \in [0, 1]$: a value of $0$ means perfectly periodic; $1$ means fully aperiodic
(noise only).

D4C estimates aperiodicity via **random frequency sub-sampling**: it evaluates the spectral envelope
at randomly offset frequency grids and measures inter-estimate variance. A periodic signal has low
variance across grids; a noisy signal has high variance. This ratio is converted into the
aperiodicity index per frequency band and frame.

---

## 5. Resynthesis: the Harmonic-plus-Noise Synthesis Model

Given the three parametric streams $(F_0[m],\, \text{sp}[m,\cdot],\, \text{ap}[m,\cdot])$, WORLD
synthesises the output waveform via a **harmonic-plus-noise model** (HNM).

At each frame $m$, the excitation is:

$$
e[m, n] =
\underbrace{\sum_{h=1}^{H[m]} A_h[m]\, \cos\!\left(2\pi h F_0[m] n / f_s + \phi_h[m]\right)}_{\text{harmonic (voiced) component}}
+\;
\underbrace{q[m, n]}_{\text{noise component}}
$$

where:
- $H[m] = \lfloor f_s / (2 F_0[m]) \rfloor$ is the number of harmonics below Nyquist,
- $A_h[m]$ is the amplitude of the $h$-th harmonic, derived from $\text{sp}$ and $\text{ap}$,
- $\phi_h[m]$ is the instantaneous phase, accumulated frame-to-frame via phase integration for continuity,
- $q[m, n]$ is coloured noise shaped by $\text{ap}[m,\cdot] \cdot \text{sp}[m,\cdot]$.

Specifically, the harmonic amplitude at frequency bin $h$ is:

$$
A_h[m] = \sqrt{\left(1 - \text{ap}[m,\, h]\right) \cdot \text{sp}[m,\, h]}
$$

and the noise power spectral density equals $\text{ap}[m, k] \cdot \text{sp}[m, k]$. The excitation
is then convolved with the minimum-phase filter whose power spectrum equals $\text{sp}[m,\cdot]$,
and successive frames are overlap-added to produce $y[n]$.

---

## 6. Pitch Shifting in the Vocoder Domain

Classical waveform-domain pitch shifting changes both pitch and formants simultaneously, producing
the well-known "chipmunk" artefact. In the vocoder domain, the two are **decoupled by construction**.

A shift of $\Delta s$ semitones is applied by multiplying all voiced $F_0$ values by:

$$
\alpha = 2^{\Delta s / 12}
$$

For each frame $m$ where $F_0[m] > 0$:

$$
F_0'[m] = \alpha \cdot F_0[m]
$$

The spectral envelope $\text{sp}[m,\cdot]$ and aperiodicity $\text{ap}[m,\cdot]$ are left
**unchanged**. Since formant positions are encoded in $\text{sp}$ independently of $F_0$, the
harmonic comb shifts while the envelope it rides under stays fixed — pitch changes, timbre does not.

---

## 7. Time-Scale Modification via Feature Warping

**Time-Scale Modification (TSM)** changes the duration of a speech signal without altering pitch or
spectral content. In the vocoder domain this is achieved by **temporal interpolation** of the
parameter frames.

Let $M$ be the original number of frames and $M'$ the target number, related by a speed factor
$\lambda > 0$:

$$
M' = \left\lfloor \frac{M}{\lambda} \right\rfloor
$$

A speed factor $\lambda > 1$ compresses the signal in time (faster speech); $\lambda < 1$ stretches
it (slower speech). Each parameter stream is resampled from $M$ to $M'$ frames by **linear
interpolation** on a normalised time grid. For a 1-D stream $\theta[m]$ (such as $F_0$), the
resampled value at new frame $m'$ is:

$$
\theta'[m'] = (1 - \alpha)\,\theta[\lfloor m \rfloor] + \alpha\,\theta[\lceil m \rceil]
\quad \text{where} \quad
m = m' \cdot \frac{M-1}{M'-1}, \quad \alpha = m - \lfloor m \rfloor
$$

For the 2-D streams $\text{sp}[m, k]$ and $\text{ap}[m, k]$, the same interpolation is applied
independently along the time axis for each frequency bin $k$.

This approach is effective because WORLD parameter streams are **slowly varying** relative to the
5 ms frame period. Linear interpolation introduces negligible artefacts for moderate speed factors
($0.5 \leq \lambda \leq 2.0$). More aggressive warping would benefit from cubic or sinc
interpolation. Crucially, the spectral envelope is also resampled (not stretched), so the formant
structure remains physically consistent throughout the modified utterance.

---

## 8. Short-Time Fourier Transform and Spectrogram

The **Short-Time Fourier Transform (STFT)** of a discrete-time signal $x[n]$ is:

$$
X[m, k] = \sum_{n=-\infty}^{+\infty} x[n]\, w[n - m H]\, e^{-j 2\pi k n / N}
$$

where $w[\cdot]$ is a Hann analysis window, $H$ is the hop size in samples, $N$ is the FFT size
(here $N = 1024$), $m$ is the frame index, and $k \in \{0, 1, \ldots, N/2\}$ is the frequency bin.

The **spectrogram** is the squared magnitude converted to decibels:

$$
P_{\text{dB}}[m, k] = 20 \log_{10}\!\left(\max\!\left(|X[m, k]|,\, \varepsilon\right)\right)
$$

where $\varepsilon = 10^{-8}$ prevents $\log(0)$. The factor $20$ (rather than $10$) is used because
$|X[m,k]|$ is an amplitude (not power) spectrum.

The **frequency resolution** of the STFT is $\Delta f = f_s / N$, and the **time resolution** per
frame is $\Delta t = H / f_s$. There is a fundamental time-frequency trade-off: increasing $N$
improves frequency resolution but reduces temporal resolution.

In the display, the $F_0$ contour extracted by WORLD is overlaid as a cyan curve. Visually, it
should track the first harmonic in the spectrogram — deviations signal estimation error or
non-stationarity.

---

## 9. Signal Pre-processing: Resampling, Trimming, Normalisation

**Resampling.** Audio from different sources may carry different sampling rates $f_s^{\text{orig}}$.
All signals are resampled to a common target rate $f_s$ (default: 24 000 Hz) using a polyphase
anti-aliasing filter. The filter applies a low-pass cutoff at $f_s/2$ to prevent aliasing, then
interpolates or decimates by ratio $r = f_s / f_s^{\text{orig}}$.

**Silence trimming.** Leading and trailing silence is removed by thresholding the short-time RMS
energy. Frames with energy below $-25\,\text{dB}$ relative to the peak are discarded, preventing
WORLD from estimating $F_0$ over silent regions (which would introduce spurious voiced frames).

**DC removal.** The sample mean $\bar{x} = \frac{1}{N}\sum_n x[n]$ is subtracted. DC offset arises
from microphone bias or DC-coupled amplifiers; it has no perceptual content and would bias
short-time energy estimators.

**Peak normalisation.** The signal is scaled to peak amplitude 0.98:

$$
x_{\text{norm}}[n] = 0.98 \cdot \frac{x[n]}{\max_n |x[n]|}
$$

This ensures consistent input dynamic range for WORLD and prevents overflow in single-precision
synthesis.

---

## 10. Waveform Similarity via Envelope Correlation

To compare the original signal $x[n]$ and the resynthesised signal $y[n]$, similarity is measured
at the **amplitude envelope** level rather than the waveform level.

**Motivation.** WORLD's resynthesis introduces independently randomised phase in the aperiodic
component. Waveform-level cross-correlation would therefore give low values even for a perfect
resynthesis, since random phase shifts decorrelate fine structure while leaving perception unchanged.
The amplitude envelope, by contrast, tracks the slow energy modulation of speech — syllabic rhythm,
prosodic contour, pause structure — and is largely invariant to fine-structure phase.

**Envelope estimation.** The amplitude envelope $\hat{e}[n]$ is estimated by full-wave rectification
followed by moving-average smoothing with a rectangular window of length
$L = f_s \cdot 20\,\text{ms}$:

$$
\hat{e}[n] = \frac{1}{L} \sum_{\ell=0}^{L-1} |x[n - \ell]|
$$

This is equivalent to low-pass filtering the rectified signal at cutoff $f_c = f_s / L \approx 50\,\text{Hz}$,
which retains syllabic amplitude modulation (typically 4-8 Hz for speech) while suppressing
pitch-level fluctuations.

**Alignment.** The resynthesised signal may differ slightly in length due to synthesis overlap-add.
It is aligned to the reference length by linear interpolation onto the same normalised time grid
used for TSM.

**Normalised correlation.** The similarity metric is the **Pearson correlation coefficient** of the
mean-subtracted envelopes:

$$
\rho = \frac{(\hat{e}_x - \bar{\hat{e}}_x)^\top (\hat{e}_y - \bar{\hat{e}}_y)}
{\|\hat{e}_x - \bar{\hat{e}}_x\|_2 \cdot \|\hat{e}_y - \bar{\hat{e}}_y\|_2}
$$

This is clipped to $[0, 1]$ and reported as a percentage. A value near 100 % indicates that the
temporal energy contour of the resynthesised signal closely matches the original. Values below ~70 %
typically indicate significant artefacts or a failure of the analysis-synthesis cycle.

---

## References

- Fant, G. (1960). *Acoustic Theory of Speech Production*. Mouton.
- Morise, M., Yokomori, F., & Ozawa, K. (2016). WORLD: a vocoder-based high-quality speech synthesis system for real-time applications. *IEICE Transactions on Information and Systems*, E99-D(7), 1877-1884.
- Kawahara, H. (2006). STRAIGHT, exploitation of the other aspect of VOCODER. *Phonetical Sciences*, 12, 21-36.
- McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis based on a sinusoidal representation. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 34(4), 744-754.
- Driedger, J., & Müller, M. (2016). A review of time-scale modification of music signals. *Applied Sciences*, 6(2), 57.