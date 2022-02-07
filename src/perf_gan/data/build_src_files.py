from os import listdir
from os.path import isfile, join
from typing import List
import pickle


def get_files_dir(path: str) -> List[str]:
    """Compute the list of the files in a given directory

    Args:
        path (str): path to the directory to look in

    Returns:
        List[str]: list of the filenames in this directory
    """
    return [f for f in listdir(path) if isfile(join(path, f))]


def export(data: dict, path: str) -> None:
    """Export data into pickle file

    Args:
        data (dict): data dictionary
        path (str): path to the file
    """
    with open(path, "wb") as file_out:
        pickle.dump(data, file_out)


midi_path = "data/midi/midi"
audio_path = "data/audio/samples"

save_midi_path = "data/midi/contours"
save_audio_path = "data/audio/contours"

midi_files = get_files_dir(midi_path)
audio_files = get_files_dir(audio_path)

print(midi_files)
print(audio_files)
