import numpy as np

from msmd.data_model.piece import Piece

import os
import math


def getNvec(spectro_index: int, performance):
    """
    Get the N hot encoded vector from MIDI matrix.
    spectro_index(int): index on spectrogram matrix that will be used to get MIDI matrix equivalent
    performance(msmd.Performance): loaded performance of the piece
    """

    midi_matrix = performance.load_midi_matrix()  # load corresponding piece's MIDI matrix
    return np.where(midi_matrix[:, spectro_index] != 0, 1,
                    0)  # converts indexes where multiple track keys overlap into 1s


def midiToPiano(midi_vector, notation: str = 'sharp') -> str:
    """
    Convert MIDI N-hot encoded vector to equivalent piano note(s).
    Requires math to be imported.
    midi_vector(numpy array): desired N-hot encoded vector to be converted
    notation(str): sharp or flat. specify which format for output to be returned in
    """

    if notation != 'sharp' and notation != 'flat': return 'Invalid notation specification'

    note_str = ''  # declaration of final returned note(s) string
    notes_sharp = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#',
                   'B']  # note array used if sharp is specified
    notes_flat = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb',
                  'B']  # note array used if flat is specified
    notes_special_sharp = ['C', 'A', 'A#', 'B']  # special note array used if sharp is specified
    notes_special_flat = ['C', 'A', 'Bb', 'B']  # special note array used if flat is specified
    key_vector88 = np.roll(midi_vector[21:109],
                           85)  # condenses the MIDI vector down to 88 keys and moves first 3 notes to the back

    for index in np.argwhere(key_vector88).flatten():  # iterate through indices where the vector is defined as 1
        note_index = index % 12  # get the note_index by modulo 12 because 12 notes in an octave
        if notation == 'sharp':  # if sharp notation specified
            # append sharp letter note
            if index > 83:
                note_str = note_str + notes_special_sharp[note_index]
            else:
                note_str = note_str + notes_sharp[note_index]
        else:  # if flat notation specified
            # append flat letter note
            if index > 83:
                note_str = note_str + notes_special_flat[note_index]
            else:
                note_str = note_str + notes_flat[note_index]

            # append octave number
        if index == 84:
            note_str = note_str + '8'  # append octave 8 to special case note
        elif index > 84:
            note_str = note_str + '0'  # append octave 0 to first 3 special case notes
        else:
            note_str = note_str + str(math.floor((index / 12) + 1))  # append octaves starting at 1

        note_str = note_str + ' '  # add space to end of note for multiple notes

    return note_str[:-1]  # remove extra space at the end


def filteredData(root_path: str) -> list:
    """
    Filter out data where performance/score isn't available.
    Requires os to be imported.
    root_path(str): path on system to msmd data set
    """
    
    files = [name for name in os.listdir(root_path) if not name.startswith('.')] #grabs all the piece names by folder
    pieces = [Piece(root=root_path, name=data) for data in files] #converts every single piece into a Piece object
    
    return [piece for piece in pieces if not len(piece.available_performances)==0] #filters out pieces that don't have available scores/performances

def getSpectrogram(spectro_index: int, performance) -> list:
    """
    Retrieves spectrogram from performance piece
    """
    maxSpecValue = 1.3246415
    spectrogram = performance.load_spectrogram() #load corresponding piece's spectogram
    image = np.reshape(spectrogram[:,spectro_index], (92, -1)) #slices spectrogram for specific index
    image *= (255/maxSpecValue) #normalize data from 0-maxSpecValue to 0-255
    image = np.clip(image.astype(int), a_min = 0, a_max = 255) #convert data into ints clipped between 0-255
    return image
