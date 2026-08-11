[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/trungtin-dinh/resynthesize_speech)

# Resynthesize Speech

This repository contains an interactive speech analysis and resynthesis mini app based on the WORLD vocoder.

The app is designed as an educational and portfolio demo for classical speech processing. It shows how a voice recording can be analysed into vocoder parameters, modified in the parametric domain, and resynthesised into a new waveform.

A Streamlit deployment is available here:

https://resynthesize-speech.streamlit.app/

## Main features

- Record a voice sample from the microphone or upload an audio file.
- Use a default speech example for quick testing.
- Analyse speech with the WORLD vocoder.
- Estimate and display the fundamental frequency contour.
- Display a spectrogram with the estimated F0 curve.
- Resynthesise the voice from WORLD parameters.
- Modify pitch in semitones.
- Modify speaking speed with a speed factor.
- Choose the target sample rate.
- Compare the original and resynthesised waveforms.
- Display an envelope-based waveform similarity score.
- Read the English and French documentation tabs.

## Method overview

The app follows a classical analysis-synthesis workflow.

First, the input audio is loaded, converted to mono, trimmed, resampled to the target sample rate, centred by removing the DC component, and peak-normalised.

Then, WORLD decomposes the speech signal into three parameter streams:

- fundamental frequency, noted F0,
- spectral envelope,
- band aperiodicity.

These parameters describe the source-filter structure of speech. The fundamental frequency represents the voiced excitation. The spectral envelope represents the vocal tract filter. The aperiodicity stream represents the noisy component of the speech signal.

The app can then modify these parameters before synthesis. Pitch shifting is performed by multiplying voiced F0 values by a semitone-dependent ratio. Time-scale modification is performed by warping the vocoder parameter streams over time.

Finally, the modified parameters are passed back to WORLD to synthesise the output waveform.

## Speech analysis

The app displays two complementary visualisations.

The spectrogram shows how the frequency content of the speech signal evolves over time. The estimated F0 contour is overlaid on top of the spectrogram, making it possible to visually inspect the pitch trajectory.

The waveform comparison shows the original and resynthesised signals in the time domain. Because vocoder synthesis may change fine phase details, the app reports an envelope-based similarity score rather than a direct waveform correlation.

## Pitch shifting

Pitch is controlled in semitones.

A shift of 12 semitones corresponds to one octave up. A shift of -12 semitones corresponds to one octave down.

In the vocoder domain, only voiced F0 frames are modified. The spectral envelope is kept unchanged, which helps preserve the speaker timbre better than naive waveform-domain pitch shifting.

## Speed modification

The speed factor controls the temporal duration of the resynthesised voice.

A value greater than 1 makes the speech faster. A value lower than 1 makes the speech slower.

This is done by interpolating the WORLD parameter streams along the time axis, instead of simply resampling the waveform. This allows duration changes while largely preserving the pitch structure.

## Repository structure

```text
.
├── app.py                 # Gradio / Hugging Face Space entry point
├── app_sl.py              # Streamlit version of the app
├── documentation_en.md    # English documentation
├── documentation_fr.md    # French documentation
├── requirements.txt       # Python dependencies
├── LICENSE.txt            # License file
└── README.md              # Repository and Hugging Face Space description
```

## Installation

Clone the repository:

```bash
git clone https://github.com/trungtin-dinh/resynthesize_speech.git
cd resynthesize_speech
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

If needed, install the main dependencies manually:

```bash
pip install gradio streamlit numpy librosa plotly pyworld soundfile
```

## Run the Gradio app

```bash
python app.py
```

The local interface will usually be available at:

```text
http://127.0.0.1:7860
```

## Run the Streamlit app

```bash
streamlit run app_sl.py
```

The local interface will usually be available at:

```text
http://localhost:8501
```

## Hugging Face Space notes

The YAML block at the top of this README is used by Hugging Face Spaces.

The current metadata launches the Gradio version:

```yaml
sdk: gradio
app_file: app.py
```

If you want Hugging Face to launch the Streamlit version instead, update the metadata to:

```yaml
sdk: streamlit
app_file: app_sl.py
```

In that case, make sure `streamlit` is included in `requirements.txt`.

## Documentation

The repository includes two Markdown documentation files:

- `documentation_en.md` for the English documentation.
- `documentation_fr.md` for the French documentation.

These files explain the speech production model, the source-filter theory, fundamental frequency and voicing, the WORLD vocoder, harmonic-plus-noise resynthesis, pitch shifting, time-scale modification, spectrogram analysis, signal preprocessing, and waveform similarity based on envelope correlation.

## Notes on audio quality

This app is intended as an educational demonstration.

Moderate pitch and speed changes usually give the most natural results. Very large pitch shifts or extreme speed factors can introduce audible artefacts because the WORLD parameters are pushed outside their usual operating range.

Short, clean voice recordings with limited background noise generally produce the best results.

## License

This project is released under the MIT License.

## Author

Developed by Trung-Tin Dinh as part of a portfolio of interactive signal, audio, image, and computer vision mini apps.
