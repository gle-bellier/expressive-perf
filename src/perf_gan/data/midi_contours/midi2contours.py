import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Load MIDI file into PrettyMIDI object


class MidiReader:

    def __init__(self, sample_len=1024, frame_rate=100):
        """Useful tool to extract contours from a MIDI file. Be careful in the choice of sample_len: it must
        be greater than the length of the longest silence.

        Args:
            sample_len (int, optional): length of the MIDI samples. Defaults to 2048.
            frame_rate (int, optional): frame_rate (number of frame by second). Defaults to 100.
        """

        self.frame_rate = frame_rate
        self.sample_len = sample_len
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

        assert len(f0) == len(lo) == len(onsets) == len(offsets) == len(
            mask
        ), f"Invalid : numbers of samples are different : f0 : {len(f0)}, lo : {len(lo)}, onsets : {len(onsets)}, offsets: {len(offsets)} , mask: {len(mask)}"

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

        # split contours into chunks for each sample
        f0 = np.split(f0, np.arange(self.sample_len, len(f0), self.sample_len))
        lo = np.split(lo, np.arange(self.sample_len, len(lo), self.sample_len))

        # we do not take into account the last chunk that has not a size equals to sample_len
        return f0[:-1], lo[:-1]

    def __get_onsets_mask(self) -> Tuple[np.ndarray]:
        """Create onsets and offsets contours and mask for the note in the track.
        We do not take into account the last chunk that has not a size equals to 
        sample_len.
        Returns:
            Tuple[np.ndarray]: onsets, offset and corresponding mask
        """

        l_onsets = []
        l_offsets = []
        l_masks = []

        l = self.sample_len
        onsets = np.zeros(l)
        offsets = np.zeros(l)
        mask = []

        i_sample = 0

        # smoothinh the mask to take into account only sustain part
        # fix attack and release to 20% of the note length
        SMOOTH_RATIO = 0.1

        for note in self.data.notes:
            m = np.ones_like(onsets)

            start = int(
                max(0,
                    note.start * self.frame_rate - i_sample * self.sample_len))
            end = int(min(len(self) - 1, note.end *
                          self.frame_rate)) - i_sample * self.sample_len

            smooth = int((end - start) * SMOOTH_RATIO)

            if start < l and end < l:
                onsets[start] = 1
                offsets[end] = 1

                # update mask
                m[:start + smooth] -= 1
                m[max(0, end - smooth):] -= 1
                mask += [m]

            elif start < l and end > l:
                onsets[start] = 1

                # update mask
                m[:start + smooth] -= 1
                mask += [m]

                # add the mask to the list of masks
                # add onsets, offsets to the lists
                l_masks += [np.array(mask)]
                l_onsets += [onsets]
                l_offsets += [offsets]

                # reset onsets, offsets and mask

                onsets = np.zeros(l)
                offsets = np.zeros(l)
                mask = []
                i_sample += 1
                # create new mask and add the end of the current note
                # if the note is longer than the sample length we crop it
                # to the end of the next sample
                offsets[min(end, len(offsets) - 1)] = 1

                m = np.ones_like(onsets)
                end -= self.sample_len
                m[max(0, end - smooth):] -= 1
                mask += [m]
                m = np.ones_like(onsets)

            else:
                #nothing to update go to next sample
                # add the mask to the list of masks
                # add onsets, offsets to the lists
                l_masks += [np.array(mask)]
                l_onsets += [onsets]
                l_offsets += [offsets]

                # reset onsets, offsets and mask

                onsets = np.zeros(l)
                offsets = np.zeros(l)
                mask = []
                i_sample += 1

        return l_onsets, l_offsets, l_masks


if __name__ == "__main__":

    m = MidiReader(sample_len=2048)
    f0, lo, onsets, offsets, mask = m.get_contours("data/midi/midi/test.mid")

    for f, l, on, off, ma in zip(f0, lo, onsets, offsets, mask):
        plt.plot(f)
        plt.plot(l)
        plt.plot(on * 150)
        plt.plot(off * (-50))
        plt.show()

        plt.imshow(ma)
        plt.show()