# Performance Modelling With GANs



Implementation of GANs for performance modelling. The different models aim at  converting MIDI files into expressive contours of fundamental frequency f_0 and loudness provided to the *DDSP* model. (https://magenta.tensorflow.org/ddsp). This last restitute the timber of a monophonic instrument and outputs the waveform corresponding to the input contours. The main idea is to generate expressive performance contours without a time-aligned dataset of musical performances and there corresponding symbolic representation (MIDI files). 
## Project structure

All models and their blocks are located in the `src/perf_gan/models` folder and the corresponding losses in the `src/perf_gan/losses`.

Run tests with *pytest* from `test/`. All data files will be written to the `data/` folder and synthetic dataset generators are located in `src/perf_gan/data`. 
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