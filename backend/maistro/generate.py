"""Generate a new MIDI composition from the trained model.

Refactored out of the original generate.py: no more module-level TF session
setup, and weight loading goes through model.load_trained_network() which
correctly finds the `.keras` checkpoints training actually produces.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from music21 import chord, instrument, note, stream

from . import config
from .model import load_trained_network

NOTES_TO_GENERATE = 300


def prepare_seed_sequences(notes: list[str], pitchnames: list[str], n_vocab: int):
    """Build all (sequence_length)-note sliding windows to seed generation from."""
    note_to_int = {n: i for i, n in enumerate(pitchnames)}
    sequence_length = config.SEQUENCE_LENGTH

    network_input = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i : i + sequence_length]
        network_input.append([note_to_int[c] for c in sequence_in])

    normalized_input = np.reshape(network_input, (len(network_input), sequence_length, 1))
    normalized_input = normalized_input / float(n_vocab)
    return network_input, normalized_input


def generate_notes(model, network_input, pitchnames: list[str], n_vocab: int) -> list[str]:
    """Autoregressively sample NOTES_TO_GENERATE tokens starting from a random seed."""
    int_to_note = {i: n for i, n in enumerate(pitchnames)}
    start = np.random.randint(0, len(network_input) - 1)
    pattern = list(network_input[start])

    prediction_output = []
    for _ in range(NOTES_TO_GENERATE):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        index = int(np.argmax(prediction))
        prediction_output.append(int_to_note[index])

        pattern.append(index)
        pattern = pattern[1:]

    return prediction_output


def convert_to_float(frac_str: str) -> float:
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split("/")
        try:
            leading, num = num.split(" ")
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac


def tokens_to_midi(prediction_output: list[str], filepath: Path) -> None:
    """Convert generated note/chord/rest tokens into a MIDI file."""
    offset = 0.0
    output_notes = []

    for token in prediction_output:
        name, duration_str = token.split()

        if ("." in name) or name.isdigit():
            notes_in_chord = name.split(".")
            chord_notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note)) if current_note.isdigit() else note.Note(current_note)
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        elif "rest" in name:
            new_rest = note.Rest()
            rest_duration = min(convert_to_float(duration_str), 4.0)
            new_rest.duration.quarterLength = rest_duration
            new_rest.offset = offset
            new_rest.storedInstrument = instrument.Piano()
            output_notes.append(new_rest)
        else:
            new_note = note.Note(name)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += convert_to_float(duration_str)

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=str(filepath))
    print(f"Generated MIDI saved as {filepath}")


def generate(output_filename: str = "generated_output.mid") -> Path:
    """Generate a new MIDI file and return its path."""
    with open(config.NOTES_FILE, "rb") as fh:
        notes = pickle.load(fh)

    pitchnames = sorted(set(notes))
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_seed_sequences(notes, pitchnames, n_vocab)
    model = load_trained_network(n_vocab)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.OUTPUT_DIR / output_filename
    tokens_to_midi(prediction_output, output_path)
    return output_path
