import music21
import numpy as np
from Config import *

def buildPitchesVocabulary(notes):
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
            piece = music21.converter.parse(midi_file)
            for element in piece.flatten().elements:
                if isinstance(element, music21.note.Rest) and element.offset != 0:
                    notes.append('R')
                if isinstance(element, music21.note.Note):
                    notes.append(str(element.pitch))
                if isinstance(element, music21.chord.Chord):
                    # print(element)
                    # print(','.join(str(pitch) for pitch in element.pitches))
                    # print()
                    notes.append(','.join(str(pitch) for pitch in element.pitches))

        return notes

def makeData(dataset_path=DATASET_PATH):
    trained_data = []
    notes = extractFeatures(dataset_path)
    notes_names = buildPitchesVocabulary(notes)
    notes_to_ints = notesNames2ints(notes_names)
    for i in range(0, len(notes) - SEQUENCE_LENGTH):
        piece = [notes_to_ints[chr] for chr in notes[i:i + SEQUENCE_LENGTH]]
        trained_data.append(piece)
    
    trained_data = np.array(trained_data)
    trained_data = (trained_data - float(len(notes)) / 2) / (float(len(notes)) / 2)
    return trained_data, notes_to_ints
    
def processChords(chords_str):
    chords = chords_str.split(',')
    notes = []
    for note_str in chords:
        note_obj = music21.note.Note(note_str)
        note_obj.storedInstrument = music21.instrument.Piano()
        notes.append(note_obj)
    return notes

def seq_to_midi(sequence, filename):
    # Generate pieces of music with well-trained generator then transform them into midi
    stream = music21.stream.Stream()

    for note in sequence:
        if note == 'R':
            stream.append(music21.note.Rest())
        elif (',' in note) or note.isdigit():
            chord = processChords(note)
            chord_obj = music21.chord.Chord(chord)
            stream.append(chord_obj)
        else:
            note_obj = music21.note.Note(note)
            note_obj.storedInstrument = music21.instrument.Piano()
            stream.append(note_obj)

    stream.write('midi', fp=f'{filename}.midi')