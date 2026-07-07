"""AI-based "collaboration" mode: extend a user-recorded MIDI performance using
the trained LSTM model, seeded from the recording's last note.

Used only by the pygame desktop UI (audio_to_spectrogram.py). Model
architecture/weight-loading now goes through backend.maistro so this no longer
duplicates that logic (and picks up the .keras weight-loading fix).
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np
from music21 import chord, converter, instrument, interval, note, pitch, stream

from backend.maistro import config
from backend.maistro.model import load_trained_network


def prepare_sequences_output(notes: list[str], pitchnames: list[str], n_vocab: int):
    note_to_int = {n: i for i, n in enumerate(pitchnames)}
    sequence_length = config.SEQUENCE_LENGTH

    network_input = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i : i + sequence_length]
        network_input.append([note_to_int[c] for c in sequence_in])

    normalized_input = np.reshape(network_input, (len(network_input), sequence_length, 1))
    normalized_input = normalized_input / float(n_vocab)
    return network_input, normalized_input


def get_seed_note_from_latest_midi(recording_path: Path) -> str:
    """Parse the recording and return its last note (or first pitch of its last chord)."""
    print(f"Using recording: {recording_path} for seed note extraction")
    midi_stream = converter.parse(str(recording_path))

    parts = instrument.partitionByInstrument(midi_stream)
    notes_to_parse = parts.parts[0].recurse() if parts else midi_stream.flat.notes

    elements = []
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            elements.append(f"{element.pitch} {element.duration.quarterLength}")
        elif isinstance(element, chord.Chord):
            elements.append(f"{element.pitches[0]} {element.duration.quarterLength}")

    if not elements:
        raise ValueError("No note elements found in the recorded MIDI file.")

    seed_note = elements[-1]
    print("Extracted seed note:", seed_note)
    return seed_note


def next_seed(seed_str: str) -> str:
    """Transpose a 'NOTE duration' seed string up by one semitone."""
    note_part, duration = seed_str.split()
    new_pitch = pitch.Pitch(note_part).transpose(interval.Interval(1))
    return f"{new_pitch.nameWithOctave} {duration}"


def generate_notes(model, network_input, pitchnames: list[str], n_vocab: int, recording_path: Path) -> list[str]:
    note_to_int = {n: i for i, n in enumerate(pitchnames)}
    int_to_note = {i: n for i, n in enumerate(pitchnames)}

    seed_note = get_seed_note_from_latest_midi(recording_path)
    max_attempts = 12
    pattern = None
    attempt = 0

    while attempt < max_attempts and pattern is None:
        if seed_note not in note_to_int:
            print(f"Seed note {seed_note} not found in training data. Trying next note.")
            seed_note = next_seed(seed_note)
            attempt += 1
            continue

        desired_seed_value = note_to_int[seed_note]
        matching_sequences = [seq.copy() for seq in network_input if seq[-1] == desired_seed_value]

        if matching_sequences:
            pattern = random.choice(matching_sequences)
            break
        print(f"No sequence ending with {seed_note} found. Trying next note.")
        seed_note = next_seed(seed_note)
        attempt += 1

    if pattern is None:
        raise ValueError("No seed sequence ending with a valid note candidate was found in network_input.")

    print("Selected seed sequence (notes):", [int_to_note[i] for i in pattern])

    prediction_output = []
    for _ in range(150):
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


def create_midi(prediction_output: list[str], recording_path: Path, output_path: Path) -> None:
    offset = 0.0
    output_notes = []
    rest_duration = 0.0

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

        offset += rest_duration if "rest" in name else convert_to_float(duration_str)

    original_stream = converter.parse(str(recording_path))
    midi_stream = stream.Stream(output_notes)

    merged_midi = stream.Stream()
    merged_midi.append(original_stream.flat)
    merged_midi.append(midi_stream.flat)

    merged_midi.write("midi", fp=str(output_path))
    print(f"Generated MIDI saved as {output_path}")


def generate_collab(
    recording_path: Path = config.PROJECT_ROOT / "recording.mid",
    output_path: Path = config.PROJECT_ROOT / "collab_output.mid",
) -> Path:
    """Extend a recorded MIDI performance with AI-generated continuation notes."""
    with open(config.NOTES_FILE, "rb") as fh:
        notes = pickle.load(fh)

    pitchnames = sorted(set(notes))
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences_output(notes, pitchnames, n_vocab)
    model = load_trained_network(n_vocab)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab, recording_path)

    create_midi(prediction_output, recording_path, output_path)
    return output_path
