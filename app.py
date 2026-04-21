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
DEFAULT_AUDIO_URL = "https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"

REFERENCE_SENTENCE = (
    "Today we measure how a voice changes in pitch, resonance, and timing during natural speech."
)

LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]

DOCUMENTATION_fr = """
---

## Table des matières

1. [Le modèle de production de la parole](#1-le-modèle-de-production-de-la-parole)
2. [La théorie source-filtre](#2-la-théorie-source-filtre)
3. [Fréquence fondamentale et voisement](#3-fréquence-fondamentale-et-voisement)
4. [Le vocodeur WORLD](#4-le-vocodeur-world)
5. [Resynthèse : le modèle harmonique-plus-bruit](#5-resynthèse-le-modèle-harmonique-plus-bruit)
6. [Transposition de hauteur dans le domaine du vocodeur](#6-transposition-de-hauteur-dans-le-domaine-du-vocodeur)
7. [Modification d'échelle temporelle par déformation des paramètres](#7-modification-déchelle-temporelle-par-déformation-des-paramètres)
8. [Transformée de Fourier à court terme et spectrogramme](#8-transformée-de-fourier-à-court-terme-et-spectrogramme)
9. [Prétraitement du signal](#9-prétraitement-du-signal-rééchantillonnage-troncature-normalisation)
10. [Similarité des formes d'onde par corrélation des enveloppes](#10-similarité-des-formes-donde-par-corrélation-des-enveloppes)

---

## 1. Le modèle de production de la parole

La parole humaine résulte d'un processus physique décomposable en trois étapes anatomiques distinctes :
**la génération du flux d'air**, **la phonation** et **l'articulation**.

Les poumons fournissent un flux d'air sous-glottique quasi stationnaire. Lorsque le voisement est actif, les plis vocaux (glotte)
oscillent périodiquement sous l'effet de la tension musculaire et des forces de Bernoulli, découpant le
flux d'air en un train d'impulsions quasi périodique. Ce train d'impulsions est ensuite mis en forme par les cavités
résonantes du conduit vocal - pharynx, cavité buccale, cavité nasale et lèvres - avant d'être rayonné vers l'extérieur.

Le conduit vocal agit comme un tube acoustique de géométrie variable dans le temps. Ses fréquences de résonance, appelées
**formants** et couramment notées $F_1, F_2, F_3, \\ldots$, sont déterminées par la configuration instantanée
des articulateurs (langue, mâchoire, voile du palais, lèvres). Ces formants portent l'identité phonétique
des voyelles et façonnent l'enveloppe spectrale des consonnes.

Cette intuition physique motive une factorisation mathématique au coeur du traitement classique de la parole.

---

## 2. La théorie source-filtre

Le modèle source-filtre, formalisé par Fant (1960), représente le signal de parole $s[n]$ comme la
sortie d'un système linéaire variant dans le temps excité par un signal source $e[n]$ :

$$
S(z) = E(z) \\cdot H(z)
$$

où $E(z)$ est la transformée en Z de l'excitation et $H(z)$ est la fonction de transfert du filtre du conduit
vocal. Dans le domaine fréquentiel, et sous l'hypothèse de stationnarité à court terme, cela
factorise le spectre de puissance à court terme :

$$
|S(e^{j\\omega})|^2 \\;\\approx\\; |E(e^{j\\omega})|^2 \\cdot |H(e^{j\\omega})|^2
$$

L'**excitation** $e[n]$ est soit :
- un **train d'impulsions périodique** à la fréquence $F_0$ pendant une phonation **voisée**, soit
- un **processus de bruit blanc** pendant les segments **non voisés** (fricatifs, aspirés).

Le **filtre** $H(e^{j\\omega})$ code la forme du conduit vocal et définit l'**enveloppe spectrale** -
une fonction lisse, variant lentement avec la fréquence. Point crucial, l'enveloppe spectrale est indépendante de
$F_0$ : une même voyelle prononcée avec des hauteurs différentes conserve la même structure de formants.

Cette indépendance est la clé qui permet une **manipulation indépendante** de la hauteur (périodicité de la source)
et du timbre (forme du filtre), ce qu'exploitent précisément les vocodeurs.

---

## 3. Fréquence fondamentale et voisement

La **fréquence fondamentale** $F_0$ (en Hz) est l'inverse de la période glottique $T_0$ :

$$
F_0 = \\frac{1}{T_0}
$$

Les valeurs typiques vont de 80 à 180 Hz pour des locuteurs masculins et de 160 à 300 Hz pour des locuteurs féminins. Exprimés
sur une **échelle musicale**, les intervalles de hauteur se mesurent en **demi-tons**. Un demi-ton correspond à un
rapport de fréquence de $2^{1/12}$, donc un décalage de $\\Delta s$ demi-tons transforme une fréquence $F_0$ en :

$$
F_0' = F_0 \\cdot 2^{\\Delta s / 12}
$$

Une octave au-dessus correspond à $\\Delta s = 12$, c'est-à-dire à un doublement de $F_0$.

La décision voisé / non voisé est tout aussi importante. Dans les trames non voisées, $F_0$ n'est pas défini - ou
est conventionnellement fixé à zéro - et l'excitation ne porte aucune structure harmonique. Toute
manipulation de hauteur doit donc agir **uniquement sur les trames voisées** (pour lesquelles $F_0 > 0$), en laissant les trames non voisées
inchangées afin de préserver le naturel des fricatives et des plosives.

---

## 4. Le vocodeur WORLD

WORLD (Morise et al., 2016) est un système d'analyse-synthèse de parole de haute qualité et à faible latence.
Il décompose le signal de parole en trois flux paramétriques, chacun échantillonné avec une
**période de trame** fixe $T_f$ (ici $T_f = 5\\,\\text{ms}$) :

| Paramètre | Symbole | Dimension par trame |
|---|---|---|
| Fréquence fondamentale | $F_0[m]$ | scalaire (Hz, ou 0 si non voisé) |
| Enveloppe spectrale | $\\text{sp}[m, k]$ | $N_\\text{fft}/2 + 1$ valeurs réelles |
| Apériodicité par bande | $\\text{ap}[m, k]$ | $N_\\text{fft}/2 + 1$ valeurs dans $[0,1]$ |

où $m$ est l'indice de trame et $k$ l'indice de bin fréquentiel.

### 4.1 Estimation de $F_0$ - DIO + StoneMask

WORLD estime $F_0$ en deux passes.

**DIO** (Distributed Inline-filter Operation) est un estimateur robuste et peu coûteux en calcul
de $F_0$. Il opère dans le domaine temporel en calculant des intervalles entre passages par zéro sur des versions filtrées passe-bande
du signal. Pour un ensemble de plages candidates de $F_0$, le signal est filtré et la
période instantanée est estimée. Le candidat présentant le meilleur score de périodicité est retenu.

**StoneMask** est une étape de raffinement qui corrige l'estimation grossière fournie par DIO via une estimation de fréquence instantanée
sur l'harmonique dominante. Pour chaque trame $m$, l'estimation raffinée est :

$$
\\hat{F}_0[m] = \\arg\\max_{F \\in \\mathcal{N}(F_0^{\\text{DIO}}[m])}
\\left\\{ \\frac{1}{H} \\sum_{h=1}^{H} \\frac{|X(m,\\, hF)|^2}{\\sigma^2_{\\text{noise}}} \\right\\}
$$

où $\\mathcal{N}(\\cdot)$ désigne un voisinage local autour de l'estimation DIO, $H$ est le nombre
d'harmoniques utilisées, et $X(m, f)$ est le spectre à court terme à la trame $m$ et à la fréquence $f$.

### 4.2 Estimation de l'enveloppe spectrale - CheapTrick

CheapTrick estime l'**enveloppe spectrale de puissance** $|H(e^{j\\omega})|^2$ d'une manière robuste
aux erreurs d'estimation de $F_0$ et qui évite les interférences spectrales dues aux harmoniques individuelles.

**Étape 1 - fenêtrage adaptatif à $F_0$.** Une fenêtre de Hanning de longueur proportionnelle à $3/F_0$ est appliquée
autour de chaque trame, de sorte que la fenêtre couvre toujours exactement 3 périodes fondamentales, ce qui adapte la
résolution spectrale à la hauteur.

**Étape 2 - lissage du spectre de puissance.** Le module au carré de la STFT fenêtrée est calculé :

$$
P[m, k] = \\left| \\sum_{n} x[n]\\, w[n - m T_f]\\, e^{-j 2\\pi k n / N} \\right|^2
$$

Une opération de liftering dans le domaine du quefrency supprime ensuite les ondulations harmoniques. Par un liftering cepstral au-dessus
du quefrency de coupure $q_c = 1/F_0$, la structure harmonique fine est retirée, ne laissant que l'enveloppe
lisse.

**Étape 3 - récupération spectrale.** Le log-spectre lissé est contraint à être le log d'un spectre
de puissance valide via une correction à phase minimale. Cette étape empêche l'apparition de valeurs spectrales négatives en
synthèse.

### 4.3 Estimation de l'apériodicité - D4C

La parole voisée naturelle n'est jamais parfaitement périodique. Le souffle, la turbulence glottique et
la coarticulation introduisent une **composante stochastique** modélisée par l'**apériodicité par bande**
$\\text{ap}[m, k] \\in [0, 1]$ : une valeur de $0$ signifie parfaitement périodique ; $1$ signifie totalement
apériodique (bruit seul).

D4C estime l'apériodicité par **sous-échantillonnage fréquentiel aléatoire** : il évalue l'enveloppe spectrale
sur des grilles fréquentielles décalées aléatoirement et mesure la variance entre les estimations. Un signal périodique présente une faible
variance entre les grilles ; un signal bruité présente une forte variance. Ce rapport est converti en
indice d'apériodicité pour chaque bande de fréquence et chaque trame.

---

## 5. Resynthèse : le modèle de synthèse harmonique-plus-bruit

Étant donnés les trois flux paramétriques $(F_0[m],\\, \\text{sp}[m,\\cdot],\\, \\text{ap}[m,\\cdot])$, WORLD
synthétise la forme d'onde de sortie au moyen d'un **modèle harmonique-plus-bruit** (HNM).

À chaque trame $m$, l'excitation est :

$$
e[m, n] =
\\underbrace{\\sum_{h=1}^{H[m]} A_h[m]\\, \\cos\\!\\left(2\\pi h F_0[m] n / f_s + \\phi_h[m]\\right)}_{\\text{composante harmonique (voisée)}}
+\\;
\\underbrace{q[m, n]}_{\\text{composante de bruit}}
$$

où :
- $H[m] = \\lfloor f_s / (2 F_0[m]) \\rfloor$ est le nombre d'harmoniques sous la fréquence de Nyquist,
- $A_h[m]$ est l'amplitude de la $h$-ième harmonique, dérivée de $\\text{sp}$ et de $\\text{ap}$,
- $\\phi_h[m]$ est la phase instantanée, accumulée de trame en trame par intégration de phase pour assurer la continuité,
- $q[m, n]$ est un bruit coloré mis en forme par $\\text{ap}[m,\\cdot] \\cdot \\text{sp}[m,\\cdot]$.

Plus précisément, l'amplitude harmonique au bin fréquentiel $h$ est :

$$
A_h[m] = \\sqrt{\\left(1 - \\text{ap}[m,\\, h]\\right) \\cdot \\text{sp}[m,\\, h]}
$$

et la densité spectrale de puissance du bruit vaut $\\text{ap}[m, k] \\cdot \\text{sp}[m, k]$. L'excitation
est ensuite convoluée avec le filtre à phase minimale dont le spectre de puissance est égal à $\\text{sp}[m,\\cdot]$,
et les trames successives sont recombinées par overlap-add pour produire $y[n]$.

---

## 6. Transposition de hauteur dans le domaine du vocodeur

La transposition de hauteur classique dans le domaine temporel modifie simultanément la hauteur et les formants, produisant
le fameux artefact de "voix de dessin animé". Dans le domaine du vocodeur, les deux sont **découplés par construction**.

Un décalage de $\\Delta s$ demi-tons est appliqué en multipliant toutes les valeurs voisées de $F_0$ par :

$$
\\alpha = 2^{\\Delta s / 12}
$$

Pour chaque trame $m$ telle que $F_0[m] > 0$ :

$$
F_0'[m] = \\alpha \\cdot F_0[m]
$$

L'enveloppe spectrale $\\text{sp}[m,\\cdot]$ et l'apériodicité $\\text{ap}[m,\\cdot]$ sont laissées
**inchangées**. Comme les positions des formants sont codées dans $\\text{sp}$ indépendamment de $F_0$, le
peigne harmonique se déplace tandis que l'enveloppe dans laquelle il s'inscrit reste fixe - la hauteur change, le timbre non.

---

## 7. Modification d'échelle temporelle par déformation des paramètres

La **modification d'échelle temporelle** (TSM) change la durée d'un signal de parole sans modifier sa hauteur ni son
contenu spectral. Dans le domaine du vocodeur, cela s'obtient par **interpolation temporelle** des
trames paramétriques.

Soit $M$ le nombre initial de trames et $M'$ le nombre cible, liés par un facteur de vitesse
$\\lambda > 0$ :

$$
M' = \\left\\lfloor \\frac{M}{\\lambda} \\right\\rfloor
$$

Un facteur de vitesse $\\lambda > 1$ compresse le signal dans le temps (parole plus rapide) ; $\\lambda < 1$ l'étire
(parole plus lente). Chaque flux paramétrique est rééchantillonné de $M$ vers $M'$ trames par **interpolation
linéaire** sur une grille temporelle normalisée. Pour un flux 1-D $\\theta[m]$ (tel que $F_0$), la
valeur rééchantillonnée à la nouvelle trame $m'$ est :

$$
\\theta'[m'] = (1 - \\alpha)\\,\\theta[\\lfloor m \\rfloor] + \\alpha\\,\\theta[\\lceil m \\rceil]
\\quad \\text{avec} \\quad
m = m' \\cdot \\frac{M-1}{M'-1}, \\quad \\alpha = m - \\lfloor m \\rfloor
$$

Pour les flux 2-D $\\text{sp}[m, k]$ et $\\text{ap}[m, k]$, la même interpolation est appliquée
indépendamment selon l'axe temporel pour chaque bin fréquentiel $k$.

Cette approche est efficace parce que les flux paramétriques de WORLD varient **lentement** relativement à la
période de trame de 5 ms. L'interpolation linéaire introduit des artefacts négligeables pour des facteurs de vitesse modérés
($0.5 \\leq \\lambda \\leq 2.0$). Des déformations plus fortes bénéficieraient d'une interpolation cubique ou sinc.
Point crucial, l'enveloppe spectrale est elle aussi rééchantillonnée (et non étirée), de sorte que la structure des formants
reste physiquement cohérente tout au long de l'énoncé modifié.

---

## 8. Transformée de Fourier à court terme et spectrogramme

La **transformée de Fourier à court terme** (STFT) d'un signal en temps discret $x[n]$ est :

$$
X[m, k] = \\sum_{n=-\\infty}^{+\\infty} x[n]\\, w[n - m H]\\, e^{-j 2\\pi k n / N}
$$

où $w[\\cdot]$ est une fenêtre d'analyse de Hann, $H$ est le pas en échantillons, $N$ est la taille
de FFT (ici $N = 1024$), $m$ est l'indice de trame, et $k \\in \\{0, 1, \\ldots, N/2\\}$ est le bin fréquentiel.

Le **spectrogramme** est le module au carré converti en décibels :

$$
P_{\\text{dB}}[m, k] = 20 \\log_{10}\\!\\left(\\max\\!\\left(|X[m, k]|,\\, \\varepsilon\\right)\\right)
$$

où $\\varepsilon = 10^{-8}$ empêche $\\log(0)$. Le facteur $20$ (plutôt que $10$) est utilisé parce que
$|X[m,k]|$ est un spectre d'amplitude (et non de puissance).

La **résolution fréquentielle** de la STFT est $\\Delta f = f_s / N$, et la **résolution temporelle** par
trame est $\\Delta t = H / f_s$. Il existe un compromis fondamental temps-fréquence : augmenter $N$
améliore la résolution fréquentielle mais dégrade la résolution temporelle.

Dans l'affichage, le contour de $F_0$ extrait par WORLD est superposé sous forme d'une courbe cyan. Visuellement, il
doit suivre la première harmonique du spectrogramme - des écarts signalent une erreur d'estimation ou une
non-stationnarité.

---

## 9. Prétraitement du signal : rééchantillonnage, troncature, normalisation

**Rééchantillonnage.** Les signaux audio issus de sources différentes peuvent avoir des fréquences d'échantillonnage $f_s^{\\text{orig}}$
différentes. Tous les signaux sont rééchantillonnés vers une fréquence cible commune $f_s$ (par défaut : 24 000 Hz) à l'aide d'un filtre
polyphasé anti-repliement. Le filtre applique une coupure passe-bas à $f_s/2$ pour éviter le repliement, puis
interpole ou décime selon le rapport $r = f_s / f_s^{\\text{orig}}$.

**Suppression des silences.** Les silences en début et en fin de signal sont retirés par seuillage de l'énergie RMS à court terme.
Les trames dont l'énergie est inférieure de $-25\\,\\text{dB}$ au pic sont supprimées, ce qui évite à
WORLD d'estimer $F_0$ sur des régions silencieuses (ce qui introduirait de fausses trames voisées).

**Suppression de la composante continue.** La moyenne des échantillons $\\bar{x} = \\frac{1}{N}\\sum_n x[n]$ est soustraite. L'offset continu provient
du biais du microphone ou d'amplificateurs couplés en continu ; il n'a aucun contenu perceptif et biaiserait les
estimateurs d'énergie à court terme.

**Normalisation de crête.** Le signal est mis à l'échelle pour avoir une amplitude de crête égale à 0.98 :

$$
x_{\\text{norm}}[n] = 0.98 \\cdot \\frac{x[n]}{\\max_n |x[n]|}
$$

Cela garantit une plage dynamique d'entrée cohérente pour WORLD et évite les débordements en synthèse
simple précision.

---

## 10. Similarité des formes d'onde par corrélation des enveloppes

Pour comparer le signal original $x[n]$ et le signal resynthétisé $y[n]$, la similarité est mesurée
au niveau de l'**enveloppe d'amplitude** plutôt qu'au niveau de la forme d'onde elle-même.

**Motivation.** La resynthèse par WORLD introduit une phase aléatoire indépendante dans la composante apériodique.
Une corrélation croisée au niveau de la forme d'onde donnerait donc de faibles valeurs même pour une resynthèse parfaite,
car les décalages de phase aléatoires décorrèlent la structure fine tout en laissant la perception inchangée.
L'enveloppe d'amplitude, au contraire, suit les modulations lentes d'énergie de la parole - rythme syllabique,
contour prosodique, structure des pauses - et est largement invariante aux phases de structure fine.

**Estimation de l'enveloppe.** L'enveloppe d'amplitude $\\hat{e}[n]$ est estimée par redressement double alternance
suivi d'un lissage par moyenne glissante avec une fenêtre rectangulaire de longueur
$L = f_s \\cdot 20\\,\\text{ms}$ :

$$
\\hat{e}[n] = \\frac{1}{L} \\sum_{\\ell=0}^{L-1} |x[n - \\ell]|
$$

Ceci équivaut à filtrer passe-bas le signal redressé à une fréquence de coupure $f_c = f_s / L \\approx 50\\,\\text{Hz}$,
ce qui conserve la modulation d'amplitude syllabique (typiquement 4-8 Hz pour la parole) tout en supprimant
les fluctuations à l'échelle de la hauteur.

**Alignement.** Le signal resynthétisé peut différer légèrement en longueur à cause de l'overlap-add en synthèse.
Il est aligné sur la longueur de référence par interpolation linéaire sur la même grille temporelle normalisée
que celle utilisée pour la modification d'échelle temporelle.

**Corrélation normalisée.** La métrique de similarité est le **coefficient de corrélation de Pearson** des
enveloppes centrées :

$$
\\rho = \\frac{(\\hat{e}_x - \\bar{\\hat{e}}_x)^\\top (\\hat{e}_y - \\bar{\\hat{e}}_y)}
{\\|\\hat{e}_x - \\bar{\\hat{e}}_x\\|_2 \\cdot \\|\\hat{e}_y - \\bar{\\hat{e}}_y\\|_2}
$$

Cette valeur est tronquée dans $[0, 1]$ et affichée en pourcentage. Une valeur proche de 100 % indique que
le contour énergétique temporel du signal resynthétisé correspond étroitement à celui de l'original. Des valeurs inférieures à environ 70 %
indiquent généralement des artefacts importants ou un échec du cycle analyse-synthèse.

---

## Références

- Fant, G. (1960). *Acoustic Theory of Speech Production*. Mouton.
- Morise, M., Yokomori, F., & Ozawa, K. (2016). WORLD: a vocoder-based high-quality speech synthesis system for real-time applications. *IEICE Transactions on Information and Systems*, E99-D(7), 1877-1884.
- Kawahara, H. (2006). STRAIGHT, exploitation of the other aspect of VOCODER. *Phonetical Sciences*, 12, 21-36.
- McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis based on a sinusoidal representation. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 34(4), 744-754.
- Driedger, J., & Müller, M. (2016). A review of time-scale modification of music signals. *Applied Sciences*, 6(2), 57.
"""

DOCUMENTATION_en = """
---

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
**formants** and commonly denoted $F_1, F_2, F_3, \\ldots$, are determined by the instantaneous
configuration of articulators (tongue, jaw, velum, lips). These formants carry the phonetic identity
of vowels and shape the spectral envelope of consonants.

This physical intuition motivates a mathematical factorisation at the heart of classical speech
processing.

---

## 2. The Source-Filter Theory

The source-filter model, formalised by Fant (1960), represents the speech signal $s[n]$ as the
output of a time-varying linear system driven by a source signal $e[n]$:

$$
S(z) = E(z) \\cdot H(z)
$$

where $E(z)$ is the Z-transform of the excitation and $H(z)$ is the transfer function of the vocal
tract filter. In the frequency domain and under the short-time stationarity assumption, this
factorises the short-time power spectrum:

$$
|S(e^{j\\omega})|^2 \\;\\approx\\; |E(e^{j\\omega})|^2 \\cdot |H(e^{j\\omega})|^2
$$

The **excitation** $e[n]$ is either:
- a **periodic pulse train** at rate $F_0$ during **voiced** phonation, or
- a **white noise process** during **unvoiced** (fricative, aspirate) segments.

The **filter** $H(e^{j\\omega})$ encodes the vocal tract shape and defines the **spectral envelope** —
a smooth, slowly-varying function of frequency. Crucially, the spectral envelope is independent of
$F_0$: the same vowel spoken at different pitches has the same formant structure.

This independence is the key that allows **independent manipulation** of pitch (source periodicity)
and timbre (filter shape), which is precisely what vocoders exploit.

---

## 3. Fundamental Frequency and Voicing

The **fundamental frequency** $F_0$ (in Hz) is the inverse of the glottal period $T_0$:

$$
F_0 = \\frac{1}{T_0}
$$

Typical values range from 80-180 Hz for male speakers and 160-300 Hz for female speakers. Expressed
on a **musical scale**, pitch intervals are measured in **semitones**. One semitone corresponds to a
frequency ratio of $2^{1/12}$, so a shift of $\\Delta s$ semitones maps a frequency $F_0$ to:

$$
F_0' = F_0 \\cdot 2^{\\Delta s / 12}
$$

An octave up corresponds to $\\Delta s = 12$, i.e. a doubling of $F_0$.

The voiced/unvoiced decision is equally important. In unvoiced frames, $F_0$ is undefined — or
conventionally set to zero — and the excitation carries no harmonic structure. Any pitch
manipulation must therefore act **only on voiced frames** (where $F_0 > 0$), leaving unvoiced
frames untouched to preserve the naturalness of fricatives and plosives.

---

## 4. The WORLD Vocoder

WORLD (Morise et al., 2016) is a high-quality, low-latency analysis-synthesis system for speech.
It decomposes the speech signal into three parametric streams, each sampled at a fixed
**frame period** $T_f$ (here $T_f = 5\\,\\text{ms}$):

| Parameter | Symbol | Dimension per frame |
|---|---|---|
| Fundamental frequency | $F_0[m]$ | scalar (Hz, or 0 if unvoiced) |
| Spectral envelope | $\\text{sp}[m, k]$ | $N_\\text{fft}/2 + 1$ real values |
| Band aperiodicity | $\\text{ap}[m, k]$ | $N_\\text{fft}/2 + 1$ values in $[0,1]$ |

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
\\hat{F}_0[m] = \\arg\\max_{F \\in \\mathcal{N}(F_0^{\\text{DIO}}[m])}
\\left\\{ \\frac{1}{H} \\sum_{h=1}^{H} \\frac{|X(m,\\, hF)|^2}{\\sigma^2_{\\text{noise}}} \\right\\}
$$

where $\\mathcal{N}(\\cdot)$ denotes a local neighbourhood around the DIO estimate, $H$ is the number
of harmonics used, and $X(m, f)$ is the short-time spectrum at frame $m$ and frequency $f$.

### 4.2 Spectral Envelope Estimation — CheapTrick

CheapTrick estimates the **power spectral envelope** $|H(e^{j\\omega})|^2$ in a way that is robust
to $F_0$ estimation errors and avoids spectral interference from individual harmonics.

**Step 1 — F0-adaptive windowing.** A Hanning window of length proportional to $3/F_0$ is applied
around each frame, ensuring the window always spans exactly 3 fundamental periods so spectral
resolution adapts to pitch.

**Step 2 — Power spectrum smoothing.** The squared magnitude of the windowed STFT is computed:

$$
P[m, k] = \\left| \\sum_{n} x[n]\\, w[n - m T_f]\\, e^{-j 2\\pi k n / N} \\right|^2
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
$\\text{ap}[m, k] \\in [0, 1]$: a value of $0$ means perfectly periodic; $1$ means fully aperiodic
(noise only).

D4C estimates aperiodicity via **random frequency sub-sampling**: it evaluates the spectral envelope
at randomly offset frequency grids and measures inter-estimate variance. A periodic signal has low
variance across grids; a noisy signal has high variance. This ratio is converted into the
aperiodicity index per frequency band and frame.

---

## 5. Resynthesis: the Harmonic-plus-Noise Synthesis Model

Given the three parametric streams $(F_0[m],\\, \\text{sp}[m,\\cdot],\\, \\text{ap}[m,\\cdot])$, WORLD
synthesises the output waveform via a **harmonic-plus-noise model** (HNM).

At each frame $m$, the excitation is:

$$
e[m, n] =
\\underbrace{\\sum_{h=1}^{H[m]} A_h[m]\\, \\cos\\!\\left(2\\pi h F_0[m] n / f_s + \\phi_h[m]\\right)}_{\\text{harmonic (voiced) component}}
+\\;
\\underbrace{q[m, n]}_{\\text{noise component}}
$$

where:
- $H[m] = \\lfloor f_s / (2 F_0[m]) \\rfloor$ is the number of harmonics below Nyquist,
- $A_h[m]$ is the amplitude of the $h$-th harmonic, derived from $\\text{sp}$ and $\\text{ap}$,
- $\\phi_h[m]$ is the instantaneous phase, accumulated frame-to-frame via phase integration for continuity,
- $q[m, n]$ is coloured noise shaped by $\\text{ap}[m,\\cdot] \\cdot \\text{sp}[m,\\cdot]$.

Specifically, the harmonic amplitude at frequency bin $h$ is:

$$
A_h[m] = \\sqrt{\\left(1 - \\text{ap}[m,\\, h]\\right) \\cdot \\text{sp}[m,\\, h]}
$$

and the noise power spectral density equals $\\text{ap}[m, k] \\cdot \\text{sp}[m, k]$. The excitation
is then convolved with the minimum-phase filter whose power spectrum equals $\\text{sp}[m,\\cdot]$,
and successive frames are overlap-added to produce $y[n]$.

---

## 6. Pitch Shifting in the Vocoder Domain

Classical waveform-domain pitch shifting changes both pitch and formants simultaneously, producing
the well-known "chipmunk" artefact. In the vocoder domain, the two are **decoupled by construction**.

A shift of $\\Delta s$ semitones is applied by multiplying all voiced $F_0$ values by:

$$
\\alpha = 2^{\\Delta s / 12}
$$

For each frame $m$ where $F_0[m] > 0$:

$$
F_0'[m] = \\alpha \\cdot F_0[m]
$$

The spectral envelope $\\text{sp}[m,\\cdot]$ and aperiodicity $\\text{ap}[m,\\cdot]$ are left
**unchanged**. Since formant positions are encoded in $\\text{sp}$ independently of $F_0$, the
harmonic comb shifts while the envelope it rides under stays fixed — pitch changes, timbre does not.

---

## 7. Time-Scale Modification via Feature Warping

**Time-Scale Modification (TSM)** changes the duration of a speech signal without altering pitch or
spectral content. In the vocoder domain this is achieved by **temporal interpolation** of the
parameter frames.

Let $M$ be the original number of frames and $M'$ the target number, related by a speed factor
$\\lambda > 0$:

$$
M' = \\left\\lfloor \\frac{M}{\\lambda} \\right\\rfloor
$$

A speed factor $\\lambda > 1$ compresses the signal in time (faster speech); $\\lambda < 1$ stretches
it (slower speech). Each parameter stream is resampled from $M$ to $M'$ frames by **linear
interpolation** on a normalised time grid. For a 1-D stream $\\theta[m]$ (such as $F_0$), the
resampled value at new frame $m'$ is:

$$
\\theta'[m'] = (1 - \\alpha)\\,\\theta[\\lfloor m \\rfloor] + \\alpha\\,\\theta[\\lceil m \\rceil]
\\quad \\text{where} \\quad
m = m' \\cdot \\frac{M-1}{M'-1}, \\quad \\alpha = m - \\lfloor m \\rfloor
$$

For the 2-D streams $\\text{sp}[m, k]$ and $\\text{ap}[m, k]$, the same interpolation is applied
independently along the time axis for each frequency bin $k$.

This approach is effective because WORLD parameter streams are **slowly varying** relative to the
5 ms frame period. Linear interpolation introduces negligible artefacts for moderate speed factors
($0.5 \\leq \\lambda \\leq 2.0$). More aggressive warping would benefit from cubic or sinc
interpolation. Crucially, the spectral envelope is also resampled (not stretched), so the formant
structure remains physically consistent throughout the modified utterance.

---

## 8. Short-Time Fourier Transform and Spectrogram

The **Short-Time Fourier Transform (STFT)** of a discrete-time signal $x[n]$ is:

$$
X[m, k] = \\sum_{n=-\\infty}^{+\\infty} x[n]\\, w[n - m H]\\, e^{-j 2\\pi k n / N}
$$

where $w[\\cdot]$ is a Hann analysis window, $H$ is the hop size in samples, $N$ is the FFT size
(here $N = 1024$), $m$ is the frame index, and $k \\in \\{0, 1, \\ldots, N/2\\}$ is the frequency bin.

The **spectrogram** is the squared magnitude converted to decibels:

$$
P_{\\text{dB}}[m, k] = 20 \\log_{10}\\!\\left(\\max\\!\\left(|X[m, k]|,\\, \\varepsilon\\right)\\right)
$$

where $\\varepsilon = 10^{-8}$ prevents $\\log(0)$. The factor $20$ (rather than $10$) is used because
$|X[m,k]|$ is an amplitude (not power) spectrum.

The **frequency resolution** of the STFT is $\\Delta f = f_s / N$, and the **time resolution** per
frame is $\\Delta t = H / f_s$. There is a fundamental time-frequency trade-off: increasing $N$
improves frequency resolution but reduces temporal resolution.

In the display, the $F_0$ contour extracted by WORLD is overlaid as a cyan curve. Visually, it
should track the first harmonic in the spectrogram — deviations signal estimation error or
non-stationarity.

---

## 9. Signal Pre-processing: Resampling, Trimming, Normalisation

**Resampling.** Audio from different sources may carry different sampling rates $f_s^{\\text{orig}}$.
All signals are resampled to a common target rate $f_s$ (default: 24 000 Hz) using a polyphase
anti-aliasing filter. The filter applies a low-pass cutoff at $f_s/2$ to prevent aliasing, then
interpolates or decimates by ratio $r = f_s / f_s^{\\text{orig}}$.

**Silence trimming.** Leading and trailing silence is removed by thresholding the short-time RMS
energy. Frames with energy below $-25\\,\\text{dB}$ relative to the peak are discarded, preventing
WORLD from estimating $F_0$ over silent regions (which would introduce spurious voiced frames).

**DC removal.** The sample mean $\\bar{x} = \\frac{1}{N}\\sum_n x[n]$ is subtracted. DC offset arises
from microphone bias or DC-coupled amplifiers; it has no perceptual content and would bias
short-time energy estimators.

**Peak normalisation.** The signal is scaled to peak amplitude 0.98:

$$
x_{\\text{norm}}[n] = 0.98 \\cdot \\frac{x[n]}{\\max_n |x[n]|}
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

**Envelope estimation.** The amplitude envelope $\\hat{e}[n]$ is estimated by full-wave rectification
followed by moving-average smoothing with a rectangular window of length
$L = f_s \\cdot 20\\,\\text{ms}$:

$$
\\hat{e}[n] = \\frac{1}{L} \\sum_{\\ell=0}^{L-1} |x[n - \\ell]|
$$

This is equivalent to low-pass filtering the rectified signal at cutoff $f_c = f_s / L \\approx 50\\,\\text{Hz}$,
which retains syllabic amplitude modulation (typically 4-8 Hz for speech) while suppressing
pitch-level fluctuations.

**Alignment.** The resynthesised signal may differ slightly in length due to synthesis overlap-add.
It is aligned to the reference length by linear interpolation onto the same normalised time grid
used for TSM.

**Normalised correlation.** The similarity metric is the **Pearson correlation coefficient** of the
mean-subtracted envelopes:

$$
\\rho = \\frac{(\\hat{e}_x - \\bar{\\hat{e}}_x)^\\top (\\hat{e}_y - \\bar{\\hat{e}}_y)}
{\\|\\hat{e}_x - \\bar{\\hat{e}}_x\\|_2 \\cdot \\|\\hat{e}_y - \\bar{\\hat{e}}_y\\|_2}
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
"""


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
    with gr.Tab("App"):
        with gr.Row():
            with gr.Column(scale=3):
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Voice input",
                    value=DEFAULT_AUDIO_URL,
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

    with gr.Tab("Documentation FR"):
        gr.Markdown(DOCUMENTATION_fr, latex_delimiters=LATEX_DELIMITERS)

    with gr.Tab("Documentation EN"):
        gr.Markdown(DOCUMENTATION_en, latex_delimiters=LATEX_DELIMITERS)


if __name__ == "__main__":
    demo.launch()
