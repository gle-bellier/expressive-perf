import pickle


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

    def merge(self, path: str) -> None:
        """Merge the audio and MIDI contours dataset files

        Args:
            path (str): path to the saving file for complete dataset
        """
        midi_contours = [c for c in self.__read_from_pickle(self.midi_path)]
        audio_contours = [c for c in self.__read_from_pickle(self.audio_path)]

        # compute the max number of complete samples we can generate
        nb_samples = min(len(audio_contours), len(midi_contours))
        for u_c, e_c in zip(midi_contours[:nb_samples],
                            audio_contours[:nb_samples]):
            data = {
                "u_f0": u_c["f0"],
                "u_lo": u_c["lo"],
                "e_f0": e_c["f0"],
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