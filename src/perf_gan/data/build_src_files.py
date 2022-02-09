from os import listdir
from os.path import isfile, join
from typing import List
import librosa as li
import pickle

from perf_gan.data.midi_contours.midi2contours import MidiReader
from perf_gan.data.audio_contours.extract_descriptors import Extractor


class Builder:
    def __init__(self, midi_path, save_midi_path, audio_path, save_audio_path):
        self.midi_path = midi_path
        self.save_midi_path = save_midi_path
        self.audio_path = audio_path
        self.save_audio_path = save_audio_path

        self.midi_reader = MidiReader()

    def __get_files_dir(self, path: str) -> List[str]:
        """Compute the list of the files in a given directory

        Args:
            path (str): path to the directory to look in

        Returns:
            List[str]: list of the filenames in this directory
        """
        return [f for f in listdir(path) if isfile(join(path, f))]

    def __export(self, data: dict, path: str) -> None:
        """Export data into pickle file

        Args:
            data (dict): data dictionary
            path (str): path to the file
        """
        with open(path, "ab+") as file_out:
            pickle.dump(data, file_out)

    def __buid_midi(self, sample_len=2048):
        # get list of MIDI files
        files = self.__get_files_dir(self.midi_path)

        midi_reader = MidiReader(sample_len=sample_len)

        for file in files:
            print("Extracting contours from : ", file)
            f0, lo, onsets, offsets, mask = midi_reader.get_contours(
                self.midi_path + '/' + file)

            # we export each samples
            for i in range(len(f0)):
                data = {
                    "f0": f0[i],
                    "lo": lo[i],
                    "onsets": onsets[i],
                    "offsets": offsets[i],
                    "mask": mask[i],
                }
                self.__export(data, self.save_midi_path)

    def __build_audio(self, sample_len=2048, sr=1600):
        # get list of audio files

        files = self.__get_files_dir(self.audio_path)

        for file in files:
            print("Extracting contours from ", file)
            audio, fs = li.load(self.audio_path + "/" + file, sr=sr)
            ext = Extractor(sr=sr)

            f0, lo = ext.select(audio, sample_len)

            # export each sample
            for i in range(len(f0)):
                data = {
                    "f0": f0[i],
                    "lo": lo[i],
                }
                self.__export(data, self.save_audio_path)

    def build(self, sample_len):
        self.__buid_midi(sample_len=sample_len)
        self.__build_audio(sample_len=sample_len)


if __name__ == "__main__":
    midi_path = "data/midi/midi"
    audio_path = "data/audio/samples"

    save_midi_path = "data/midi/contours/midi_contours.pickle"
    save_audio_path = "data/audio/contours/audio_contours.pickle"

    b = Builder(midi_path, save_midi_path, audio_path, save_audio_path)
    b.build(1024)
