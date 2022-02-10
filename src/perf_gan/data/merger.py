import pickle
import numpy as np
from sklearn.preprocessing import QuantileTransformer


class Merger:
    """Merge the MIDI contours dataset files and the audio contours dataset
    """
    def __init__(self, midi_path, audio_path):

        self.midi_path = midi_path
        self.audio_path = audio_path

    def __read_from_pickle(self, path):
        with open(path, 'rb') as file:
            try:
                while True:
                    yield pickle.load(file)
            except EOFError:
                pass

    def __hz2midi(self, f):
        return 12 * np.log(f / 440) + 69

    def __midi2db(self, midi_lo_sample, scaler_midi, scaler_db):

        # map midi distribution to normal distribution
        midi_sample_normal = scaler_midi.transform(
            midi_lo_sample.reshape(-1, 1))
        # apply inverse transform to map normal distribution to db distribution
        db_sample = scaler_db.transform(midi_sample_normal).squeeze()

        print("db_sample size ", db_sample.shape)

        return db_sample

    def merge(self, path: str, verbose=True) -> None:
        """Merge the audio and MIDI contours dataset files

        Args:
            path (str): path to the saving file for complete dataset
            verbose (bool, optional): if True print the steps files merging. Defaults to True.
        """
        midi_contours = [c for c in self.__read_from_pickle(self.midi_path)]
        audio_contours = [c for c in self.__read_from_pickle(self.audio_path)]

        midi_lo = np.concatenate([c["lo"]
                                  for c in midi_contours]).reshape(-1, 1)
        db_lo = np.concatenate([c["lo"]
                                for c in audio_contours]).reshape(-1, 1)

        # fit scaler to  midi distribution
        qt_midi = QuantileTransformer(n_quantiles=128)
        scaler_midi = qt_midi.fit(midi_lo)

        # fit scaler to the db distribution
        qt_db = QuantileTransformer(n_quantiles=128)
        scaler_db = qt_db.fit(db_lo)

        # compute the max number of complete samples we can generate
        nb_samples = min(len(audio_contours), len(midi_contours))
        if verbose:
            print(f"Audio dataset contains {len(audio_contours)} samples")
            print(f"Midi dataset contains {len(midi_contours)} samples")
            print(f"Merged dataset contains {nb_samples} samples")
        for u_c, e_c in zip(midi_contours[:nb_samples],
                            audio_contours[:nb_samples]):
            data = {
                "u_f0": u_c["f0"],
                "u_lo": self.__midi2db(u_c["lo"], scaler_midi, scaler_db),
                "e_f0": self.__hz2midi(e_c["f0"]),
                "e_lo": e_c["lo"],
                "onsets": u_c["onsets"],
                "offsets": u_c["offsets"],
                "mask": u_c["mask"]
            }
            with open(path, "ab+") as file_out:
                pickle.dump(data, file_out)


def main():

    audio_path = "data/audio/contours/audio_contours.pickle"
    midi_path = "data/midi/contours/midi_contours.pickle"
    m = Merger(midi_path, audio_path)

    m.merge("data/dataset.pickle")


if __name__ == '__main__':
    main()