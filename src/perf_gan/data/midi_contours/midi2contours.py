import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Load MIDI file into PrettyMIDI object


class MidiReader:
    def __init__(self, frame_rate=100):
        """Useful tool to extract contours from a MIDI file

        Args:
            frame_rate (int, optional): frame_rate (number of frame by second). Defaults to 100.
        """

        self.frame_rate = frame_rate
        self.data = None

    def __len__(self):
        return int(self.midi.get_end_time() * self.frame_rate)

    def get_contours(self, path: str) -> Tuple[np.ndarray]:
        """Compute the contours for a MIDI file, given its path.

        Args:
            path (str): path to the MIDI file

        Returns:
            Tuple[np.ndarray]: pitch, loudness, onsets, offsets and mask arrays
        """

        self.midi = pretty_midi.PrettyMIDI(path)
        self.data = self.midi.instruments[0]

        notes = self.data.get_piano_roll(self.frame_rate)
        f0, lo = self.__extract_f0_lo(notes)
        onsets, offsets, mask = self.__get_onsets_mask()

        assert (f0.shape == lo.shape == onsets.shape == offsets.shape ==
                mask[0].shape), "All contours must have the same shape"

        return f0, lo, onsets, offsets, mask

    def __extract_f0_lo(self, notes) -> Tuple[np.ndarray]:
        """Extract pitch and loudness in the piano roll for each frame

        Args:
            notes ([type]): track piano roll

        Returns:
            Tuple[np.ndarray]: pitch and loudness arrays
        """
        f0 = np.argmax(notes, axis=0)
        lo = np.transpose(np.max(notes, axis=0))
        return f0, lo

    def __get_onsets_mask(self) -> Tuple[np.ndarray]:
        """Create onsets and offsets contours and mask for the note in the track


        Returns:
            Tuple[np.ndarray]: onsets, offset and corresponding mask
        """

        onsets = np.zeros(len(self))
        offsets = np.zeros(len(self))
        mask = []

        for note in self.data.notes:
            start = int(note.start * self.frame_rate)
            end = int(min(len(self) - 1, note.end * self.frame_rate))

            onsets[start] = 1
            offsets[end] = 1

            # create one note mask and add it to the list
            m = np.ones_like(onsets)
            m[:start] -= 1
            m[end:] -= 1
            mask += [m]

        return onsets, offsets, np.array(mask)


if __name__ == "__main__":

    m = MidiReader()
    m.get_contours("data/midi/test.mid")