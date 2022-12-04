import music21
import numpy as np
from Config import *

def getNotesNames(notes):
    return sorted(set(note for note in notes))

def notesNames2ints(notes_names):
    return dict((note, number) for number, note in enumerate(notes_names))

def ints2NotesNames(notes_names):
    return dict((number, note) for number, note in enumerate(notes_names))

def extractFeatures(dataset_path=DATASET_PATH):
        import glob
        midi_files =  glob.glob(f"{dataset_path}/*.midi")
        notes = []
        for midi_file in midi_files:
            midi = music21.converter.parse(midi_file) # stream.score
            # print("Parsing %s" % midi_file)
            for element in midi.flat.elements:
                if isinstance(element, music21.note.Rest) and element.offset != 0:
                    notes.append('R')
                if isinstance(element, music21.note.Note):
                    notes.append(str(element.pitch))
                if isinstance(element, music21.chord.Chord):
                    notes.append('.'.join(str(pitch) for pitch in element.pitches))

        return notes

def makeData(dataset_path=DATASET_PATH):
    trained_data = []
    notes = extractFeatures(dataset_path)
    notes_names = getNotesNames(notes)
    notes_to_ints = notesNames2ints(notes_names)
    for i in range(0, len(notes) - SEQUENCE_LENGTH):
        trained_data.append([notes_to_ints[char] for char in notes[i:i + SEQUENCE_LENGTH]])
    
    trained_data = np.array(trained_data)
    trained_data = (trained_data - float(len(notes)) / 2) / (float(len(notes)) / 2)
    return trained_data, notes_to_ints
    
def processChords(chords_str):
    chords = chords_str.split('.')
    notes = []
    for note_str in chords:
        note_obj = music21.note.Note(note_str)
        note_obj.storedInstrument = music21.instrument.Violin()
        notes.append(note_obj)
    return notes
def seq_to_midi(sequence, filename):
    # Generate pieces of music with well-trained generator then transform them into midi
    offset = 0
    stream = music21.stream.Stream()

    for note in sequence:
        if note == 'R':
            stream.append(music21.note.Rest())
        elif ('.' in note) or note.isdigit():
            chord = processChords(note)
            chord_obj = music21.chord.Chord(chord)
            # chord_obj.offset = offset
            stream.append(chord_obj)
        # note
        else:
            note_obj = music21.note.Note(note)
            # new_note.offset = offset
            note_obj.storedInstrument = music21.instrument.Violin()
            stream.append(note_obj)

        # offset += 0.5

    # midi_stream.show('text')
    stream.write('midi', fp=f'{filename}.midi')