# Performance Modelling With GANs



Implementation of GANs for performance modelling. The different models aim at  converting MIDI files into expressive contours of fundamental frequency f_0 and loudness provided to the *DDSP* model. (https://magenta.tensorflow.org/ddsp). This last restitute the timber of a monophonic instrument and outputs the waveform corresponding to the input contours. The main idea is to generate expressive performance contours without a time-aligned dataset of musical performances and there corresponding symbolic representation (MIDI files). 
## Project structure

All models and their blocks are located in the `src/perf_gan/models` folder and the corresponding losses in the `src/perf_gan/losses`.

Run tests with *pytest* from `test/`. All data files will be written to the `data/` folder and synthetic dataset generators are located in `src/perf_gan/data`. 

## Dataset

Dataset is composed of samples of length 1024 (approximately 10s with sampling rate 100Hz.). It includes:
- from MIDI file:  
  - Pitch contour (pitch values range is 0-127 according to the MIDI norm) 
  - Loudness contour. The loudness is computed from the MIDI velocity but pre-processed such that its range matches the loudness range used ( ~ [-8, -3]).
  - Onsets of the MIDI notes
  - Offsets of the MIDI notes
  - Masks : matrix where each row corresponds to the activation of a particular note: _i.e_ equals 0 when the note is off and 1 when the note is on.
- from audio file:
  - Pitch contour, computed with CREPE model for fundamental frequency estimation but values are remapped to the MIDI range 0-127.
  - Loudness contour extracted from audio file and preprocessed with A-weightning (range is about [-8, -3]).


## Installation

1. Clone this repository:

```bash
git clone https://github.com/gle-bellier/expressive-perf.git

```

2. Install requirements:

```bash
cd expressive-perf
pip install -r requirements.txt

```