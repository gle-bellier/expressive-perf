import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
# Load MIDI file into PrettyMIDI object


class MidiReader:
    def __init__(self, path: str):
        self.midi_data = pretty_midi.PrettyMIDI(path)

    def get_f0_lo(self, frame_rate=100):

        data = self.midi_data.instruments[0]
        notes = data.get_piano_roll(frame_rate)
        f0, lo = self.__extract_f0_lo(notes)

        return f0, lo

    def __extract_f0_lo(self, notes):
        f0 = np.argmax(notes, axis=0)
        lo = np.transpose(np.max(notes, axis=0))
        return f0, lo


# if __name__ == '__main__':
#     path = "data/midi/test.mid"
#     m = MidiReader(path)
#     f0, lo = m.get_f0_lo()

#     print(f"f0 shape {f0.shape}")

#     plt.plot(f0)
#     plt.plot(lo)

#     plt.show()
