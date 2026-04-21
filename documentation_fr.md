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
**formants** et couramment notées $F_1, F_2, F_3, \ldots$, sont déterminées par la configuration instantanée
des articulateurs (langue, mâchoire, voile du palais, lèvres). Ces formants portent l'identité phonétique
des voyelles et façonnent l'enveloppe spectrale des consonnes.

Cette intuition physique motive une factorisation mathématique au coeur du traitement classique de la parole.

---

## 2. La théorie source-filtre

Le modèle source-filtre, formalisé par Fant (1960), représente le signal de parole $s[n]$ comme la
sortie d'un système linéaire variant dans le temps excité par un signal source $e[n]$ :

$$
S(z) = E(z) \cdot H(z)
$$

où $E(z)$ est la transformée en Z de l'excitation et $H(z)$ est la fonction de transfert du filtre du conduit
vocal. Dans le domaine fréquentiel, et sous l'hypothèse de stationnarité à court terme, cela
factorise le spectre de puissance à court terme :

$$
|S(e^{j\omega})|^2 \;\approx\; |E(e^{j\omega})|^2 \cdot |H(e^{j\omega})|^2
$$

L'**excitation** $e[n]$ est soit :
- un **train d'impulsions périodique** à la fréquence $F_0$ pendant une phonation **voisée**, soit
- un **processus de bruit blanc** pendant les segments **non voisés** (fricatifs, aspirés).

Le **filtre** $H(e^{j\omega})$ code la forme du conduit vocal et définit l'**enveloppe spectrale** -
une fonction lisse, variant lentement avec la fréquence. Point crucial, l'enveloppe spectrale est indépendante de
$F_0$ : une même voyelle prononcée avec des hauteurs différentes conserve la même structure de formants.

Cette indépendance est la clé qui permet une **manipulation indépendante** de la hauteur (périodicité de la source)
et du timbre (forme du filtre), ce qu'exploitent précisément les vocodeurs.

---

## 3. Fréquence fondamentale et voisement

La **fréquence fondamentale** $F_0$ (en Hz) est l'inverse de la période glottique $T_0$ :

$$
F_0 = \frac{1}{T_0}
$$

Les valeurs typiques vont de 80 à 180 Hz pour des locuteurs masculins et de 160 à 300 Hz pour des locuteurs féminins. Exprimés
sur une **échelle musicale**, les intervalles de hauteur se mesurent en **demi-tons**. Un demi-ton correspond à un
rapport de fréquence de $2^{1/12}$, donc un décalage de $\Delta s$ demi-tons transforme une fréquence $F_0$ en :

$$
F_0' = F_0 \cdot 2^{\Delta s / 12}
$$

Une octave au-dessus correspond à $\Delta s = 12$, c'est-à-dire à un doublement de $F_0$.

La décision voisé / non voisé est tout aussi importante. Dans les trames non voisées, $F_0$ n'est pas défini - ou
est conventionnellement fixé à zéro - et l'excitation ne porte aucune structure harmonique. Toute
manipulation de hauteur doit donc agir **uniquement sur les trames voisées** (pour lesquelles $F_0 > 0$), en laissant les trames non voisées
inchangées afin de préserver le naturel des fricatives et des plosives.

---

## 4. Le vocodeur WORLD

WORLD (Morise et al., 2016) est un système d'analyse-synthèse de parole de haute qualité et à faible latence.
Il décompose le signal de parole en trois flux paramétriques, chacun échantillonné avec une
**période de trame** fixe $T_f$ (ici $T_f = 5\,\text{ms}$) :

| Paramètre | Symbole | Dimension par trame |
|---|---|---|
| Fréquence fondamentale | $F_0[m]$ | scalaire (Hz, ou 0 si non voisé) |
| Enveloppe spectrale | $\text{sp}[m, k]$ | $N_\text{fft}/2 + 1$ valeurs réelles |
| Apériodicité par bande | $\text{ap}[m, k]$ | $N_\text{fft}/2 + 1$ valeurs dans $[0,1]$ |

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
\hat{F}_0[m] = \arg\max_{F \in \mathcal{N}(F_0^{\text{DIO}}[m])}
\left\{ \frac{1}{H} \sum_{h=1}^{H} \frac{|X(m,\, hF)|^2}{\sigma^2_{\text{noise}}} \right\}
$$

où $\mathcal{N}(\cdot)$ désigne un voisinage local autour de l'estimation DIO, $H$ est le nombre
d'harmoniques utilisées, et $X(m, f)$ est le spectre à court terme à la trame $m$ et à la fréquence $f$.

### 4.2 Estimation de l'enveloppe spectrale - CheapTrick

CheapTrick estime l'**enveloppe spectrale de puissance** $|H(e^{j\omega})|^2$ d'une manière robuste
aux erreurs d'estimation de $F_0$ et qui évite les interférences spectrales dues aux harmoniques individuelles.

**Étape 1 - fenêtrage adaptatif à $F_0$.** Une fenêtre de Hanning de longueur proportionnelle à $3/F_0$ est appliquée
autour de chaque trame, de sorte que la fenêtre couvre toujours exactement 3 périodes fondamentales, ce qui adapte la
résolution spectrale à la hauteur.

**Étape 2 - lissage du spectre de puissance.** Le module au carré de la STFT fenêtrée est calculé :

$$
P[m, k] = \left| \sum_{n} x[n]\, w[n - m T_f]\, e^{-j 2\pi k n / N} \right|^2
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
$\text{ap}[m, k] \in [0, 1]$ : une valeur de $0$ signifie parfaitement périodique ; $1$ signifie totalement
apériodique (bruit seul).

D4C estime l'apériodicité par **sous-échantillonnage fréquentiel aléatoire** : il évalue l'enveloppe spectrale
sur des grilles fréquentielles décalées aléatoirement et mesure la variance entre les estimations. Un signal périodique présente une faible
variance entre les grilles ; un signal bruité présente une forte variance. Ce rapport est converti en
indice d'apériodicité pour chaque bande de fréquence et chaque trame.

---

## 5. Resynthèse : le modèle de synthèse harmonique-plus-bruit

Étant donnés les trois flux paramétriques $(F_0[m],\, \text{sp}[m,\cdot],\, \text{ap}[m,\cdot])$, WORLD
synthétise la forme d'onde de sortie au moyen d'un **modèle harmonique-plus-bruit** (HNM).

À chaque trame $m$, l'excitation est :

$$
e[m, n] =
\underbrace{\sum_{h=1}^{H[m]} A_h[m]\, \cos\!\left(2\pi h F_0[m] n / f_s + \phi_h[m]\right)}_{\text{composante harmonique (voisée)}}
+\;
\underbrace{q[m, n]}_{\text{composante de bruit}}
$$

où :
- $H[m] = \lfloor f_s / (2 F_0[m]) \rfloor$ est le nombre d'harmoniques sous la fréquence de Nyquist,
- $A_h[m]$ est l'amplitude de la $h$-ième harmonique, dérivée de $\text{sp}$ et de $\text{ap}$,
- $\phi_h[m]$ est la phase instantanée, accumulée de trame en trame par intégration de phase pour assurer la continuité,
- $q[m, n]$ est un bruit coloré mis en forme par $\text{ap}[m,\cdot] \cdot \text{sp}[m,\cdot]$.

Plus précisément, l'amplitude harmonique au bin fréquentiel $h$ est :

$$
A_h[m] = \sqrt{\left(1 - \text{ap}[m,\, h]\right) \cdot \text{sp}[m,\, h]}
$$

et la densité spectrale de puissance du bruit vaut $\text{ap}[m, k] \cdot \text{sp}[m, k]$. L'excitation
est ensuite convoluée avec le filtre à phase minimale dont le spectre de puissance est égal à $\text{sp}[m,\cdot]$,
et les trames successives sont recombinées par overlap-add pour produire $y[n]$.

---

## 6. Transposition de hauteur dans le domaine du vocodeur

La transposition de hauteur classique dans le domaine temporel modifie simultanément la hauteur et les formants, produisant
le fameux artefact de "voix de dessin animé". Dans le domaine du vocodeur, les deux sont **découplés par construction**.

Un décalage de $\Delta s$ demi-tons est appliqué en multipliant toutes les valeurs voisées de $F_0$ par :

$$
\alpha = 2^{\Delta s / 12}
$$

Pour chaque trame $m$ telle que $F_0[m] > 0$ :

$$
F_0'[m] = \alpha \cdot F_0[m]
$$

L'enveloppe spectrale $\text{sp}[m,\cdot]$ et l'apériodicité $\text{ap}[m,\cdot]$ sont laissées
**inchangées**. Comme les positions des formants sont codées dans $\text{sp}$ indépendamment de $F_0$, le
peigne harmonique se déplace tandis que l'enveloppe dans laquelle il s'inscrit reste fixe - la hauteur change, le timbre non.

---

## 7. Modification d'échelle temporelle par déformation des paramètres

La **modification d'échelle temporelle** (TSM) change la durée d'un signal de parole sans modifier sa hauteur ni son
contenu spectral. Dans le domaine du vocodeur, cela s'obtient par **interpolation temporelle** des
trames paramétriques.

Soit $M$ le nombre initial de trames et $M'$ le nombre cible, liés par un facteur de vitesse
$\lambda > 0$ :

$$
M' = \left\lfloor \frac{M}{\lambda} \right\rfloor
$$

Un facteur de vitesse $\lambda > 1$ compresse le signal dans le temps (parole plus rapide) ; $\lambda < 1$ l'étire
(parole plus lente). Chaque flux paramétrique est rééchantillonné de $M$ vers $M'$ trames par **interpolation
linéaire** sur une grille temporelle normalisée. Pour un flux 1-D $\theta[m]$ (tel que $F_0$), la
valeur rééchantillonnée à la nouvelle trame $m'$ est :

$$
\theta'[m'] = (1 - \alpha)\,\theta[\lfloor m \rfloor] + \alpha\,\theta[\lceil m \rceil]
\quad \text{avec} \quad
m = m' \cdot \frac{M-1}{M'-1}, \quad \alpha = m - \lfloor m \rfloor
$$

Pour les flux 2-D $\text{sp}[m, k]$ et $\text{ap}[m, k]$, la même interpolation est appliquée
indépendamment selon l'axe temporel pour chaque bin fréquentiel $k$.

Cette approche est efficace parce que les flux paramétriques de WORLD varient **lentement** relativement à la
période de trame de 5 ms. L'interpolation linéaire introduit des artefacts négligeables pour des facteurs de vitesse modérés
($0.5 \leq \lambda \leq 2.0$). Des déformations plus fortes bénéficieraient d'une interpolation cubique ou sinc.
Point crucial, l'enveloppe spectrale est elle aussi rééchantillonnée (et non étirée), de sorte que la structure des formants
reste physiquement cohérente tout au long de l'énoncé modifié.

---

## 8. Transformée de Fourier à court terme et spectrogramme

La **transformée de Fourier à court terme** (STFT) d'un signal en temps discret $x[n]$ est :

$$
X[m, k] = \sum_{n=-\infty}^{+\infty} x[n]\, w[n - m H]\, e^{-j 2\pi k n / N}
$$

où $w[\cdot]$ est une fenêtre d'analyse de Hann, $H$ est le pas en échantillons, $N$ est la taille
de FFT (ici $N = 1024$), $m$ est l'indice de trame, et $k \in \{0, 1, \ldots, N/2\}$ est le bin fréquentiel.

Le **spectrogramme** est le module au carré converti en décibels :

$$
P_{\text{dB}}[m, k] = 20 \log_{10}\!\left(\max\!\left(|X[m, k]|,\, \varepsilon\right)\right)
$$

où $\varepsilon = 10^{-8}$ empêche $\log(0)$. Le facteur $20$ (plutôt que $10$) est utilisé parce que
$|X[m,k]|$ est un spectre d'amplitude (et non de puissance).

La **résolution fréquentielle** de la STFT est $\Delta f = f_s / N$, et la **résolution temporelle** par
trame est $\Delta t = H / f_s$. Il existe un compromis fondamental temps-fréquence : augmenter $N$
améliore la résolution fréquentielle mais dégrade la résolution temporelle.

Dans l'affichage, le contour de $F_0$ extrait par WORLD est superposé sous forme d'une courbe cyan. Visuellement, il
doit suivre la première harmonique du spectrogramme - des écarts signalent une erreur d'estimation ou une
non-stationnarité.

---

## 9. Prétraitement du signal : rééchantillonnage, troncature, normalisation

**Rééchantillonnage.** Les signaux audio issus de sources différentes peuvent avoir des fréquences d'échantillonnage $f_s^{\text{orig}}$
différentes. Tous les signaux sont rééchantillonnés vers une fréquence cible commune $f_s$ (par défaut : 24 000 Hz) à l'aide d'un filtre
polyphasé anti-repliement. Le filtre applique une coupure passe-bas à $f_s/2$ pour éviter le repliement, puis
interpole ou décime selon le rapport $r = f_s / f_s^{\text{orig}}$.

**Suppression des silences.** Les silences en début et en fin de signal sont retirés par seuillage de l'énergie RMS à court terme.
Les trames dont l'énergie est inférieure de $-25\,\text{dB}$ au pic sont supprimées, ce qui évite à
WORLD d'estimer $F_0$ sur des régions silencieuses (ce qui introduirait de fausses trames voisées).

**Suppression de la composante continue.** La moyenne des échantillons $\bar{x} = \frac{1}{N}\sum_n x[n]$ est soustraite. L'offset continu provient
du biais du microphone ou d'amplificateurs couplés en continu ; il n'a aucun contenu perceptif et biaiserait les
estimateurs d'énergie à court terme.

**Normalisation de crête.** Le signal est mis à l'échelle pour avoir une amplitude de crête égale à 0.98 :

$$
x_{\text{norm}}[n] = 0.98 \cdot \frac{x[n]}{\max_n |x[n]|}
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

**Estimation de l'enveloppe.** L'enveloppe d'amplitude $\hat{e}[n]$ est estimée par redressement double alternance
suivi d'un lissage par moyenne glissante avec une fenêtre rectangulaire de longueur
$L = f_s \cdot 20\,\text{ms}$ :

$$
\hat{e}[n] = \frac{1}{L} \sum_{\ell=0}^{L-1} |x[n - \ell]|
$$

Ceci équivaut à filtrer passe-bas le signal redressé à une fréquence de coupure $f_c = f_s / L \approx 50\,\text{Hz}$,
ce qui conserve la modulation d'amplitude syllabique (typiquement 4-8 Hz pour la parole) tout en supprimant
les fluctuations à l'échelle de la hauteur.

**Alignement.** Le signal resynthétisé peut différer légèrement en longueur à cause de l'overlap-add en synthèse.
Il est aligné sur la longueur de référence par interpolation linéaire sur la même grille temporelle normalisée
que celle utilisée pour la modification d'échelle temporelle.

**Corrélation normalisée.** La métrique de similarité est le **coefficient de corrélation de Pearson** des
enveloppes centrées :

$$
\rho = \frac{(\hat{e}_x - \bar{\hat{e}}_x)^\top (\hat{e}_y - \bar{\hat{e}}_y)}
{\|\hat{e}_x - \bar{\hat{e}}_x\|_2 \cdot \|\hat{e}_y - \bar{\hat{e}}_y\|_2}
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