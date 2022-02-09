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

    def merge(self):
        audio_contours = [c for c in self.__read_from_pickle(self.audio_path)]
        midi_contours = [c for c in self.__read_from_pickle(self.midi_path)]
        print(midi_contours[0])

        print(len(midi_contours))


def main():

    audio_path = "data/audio/contours/audio_contours.pickle"
    midi_path = "data/midi/contours/midi_contours.pickle"
    m = Merger(midi_path, audio_path)

    m.merge()


if __name__ == '__main__':
    main()