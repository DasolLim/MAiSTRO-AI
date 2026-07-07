"""Turn a folder of MIDI files into the note/chord/rest sequence used for training.

Refactored out of the original train.py (which ran this at import time) and
midi_dataset_processing.py (optional multi-track flattening helper).
"""

from __future__ import annotations

import os
import pickle
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from music21 import chord, converter, instrument, note

from . import config

MOST_COMMON_NOTE_CAP = 15000


@dataclass
class DatasetStats:
    midi_file_count: int
    note_count: int
    vocab_size: int


def convert_duration(duration_value) -> float:
    """Convert a music21 duration (possibly a fraction like '1/3') to a float."""
    try:
        return float(duration_value)
    except ValueError:
        return float(Fraction(duration_value))


def adjust_octave(pitch_name: str, shift: int = -1) -> str:
    """Shift a note's octave, e.g. to correct for music21/MIDI octave offset quirks."""
    if pitch_name[-1].isdigit():
        note_part = pitch_name[:-1]
        octave_part = int(pitch_name[-1])
        return f"{note_part}{octave_part + shift}"
    return pitch_name


def condense_multitrack(
    dataset_dir: Path = config.DATASET_DIR,
    output_dir: Path | None = None,
    program_number: int = 0,
) -> list[Path]:
    """Flatten multi-track/multi-instrument MIDI files into a single track.

    Optional preprocessing step for datasets where source files contain multiple
    instruments; writes converted copies into `output_dir` (default: dataset_dir/_condensed).
    """
    import mido
    from mido import Message, MidiFile, MidiTrack

    output_dir = output_dir or (dataset_dir / "_condensed")
    output_dir.mkdir(parents=True, exist_ok=True)

    converted: list[Path] = []
    for file in sorted(os.listdir(dataset_dir)):
        if not (file.endswith(".mid") or file.endswith(".midi")):
            continue

        midi = MidiFile(str(dataset_dir / file))
        new_midi = MidiFile()
        new_track = MidiTrack()
        new_midi.tracks.append(new_track)
        new_midi.ticks_per_beat = midi.ticks_per_beat

        events = []
        for track in midi.tracks:
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if msg.type in ("note_on", "note_off", "control_change"):
                    events.append((abs_time, msg))
        events.sort(key=lambda item: item[0])

        last_time = 0
        new_track.append(Message("program_change", program=program_number, time=0))
        for abs_time, msg in events:
            msg.time = abs_time - last_time
            new_track.append(msg)
            last_time = abs_time

        output_path = output_dir / file
        new_midi.save(str(output_path))
        converted.append(output_path)

    return converted


def extract_notes(dataset_dir: Path = config.DATASET_DIR, force: bool = False) -> list[str]:
    """Parse every MIDI file in dataset_dir into a flat list of note/chord/rest tokens.

    Results are cached to config.NOTES_FILE; pass force=True to re-parse from scratch.
    """
    if config.NOTES_FILE.exists() and not force:
        with open(config.NOTES_FILE, "rb") as fh:
            return pickle.load(fh)

    midi_files = sorted(dataset_dir.glob("*.mid"))
    if not midi_files:
        raise FileNotFoundError(f"No .mid files found in {dataset_dir}")

    notes: list[str] = []
    last_offset = 0.0

    for file in midi_files:
        try:
            midi = converter.parse(str(file))
            print(f"Parsing {file} ...")

            try:
                parts = instrument.partitionByInstrument(midi)
                notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
            except Exception:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                duration_value = convert_duration(element.quarterLength)

                if element.offset > last_offset:
                    rest_duration = element.offset - last_offset
                    if rest_duration >= 0.25:
                        notes.append(f"rest {rest_duration}")

                if isinstance(element, note.Note):
                    fixed_note = adjust_octave(element.nameWithOctave, shift=-1)
                    notes.append(f"{fixed_note} {duration_value}")
                elif isinstance(element, chord.Chord):
                    chord_notes = ".".join(
                        adjust_octave(p.nameWithOctave, shift=-1) for p in element.pitches
                    )
                    notes.append(f"{chord_notes} {duration_value}")

                last_offset = element.offset + duration_value

        except Exception as exc:
            print(f"Error processing {file}: {exc}")
            continue

    note_counts = Counter(notes)
    most_common = {n for n, _ in note_counts.most_common(MOST_COMMON_NOTE_CAP)}
    filtered_notes = [n for n in notes if n in most_common]

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.NOTES_FILE, "wb") as fh:
        pickle.dump(filtered_notes, fh)

    print(f"Extracted {len(filtered_notes)} elements from {len(midi_files)} MIDI files.")
    return filtered_notes


def prepare_dataset(dataset_dir: Path = config.DATASET_DIR, force: bool = False) -> DatasetStats:
    """Full "prepare dataset" step exposed to the API: parse MIDI files, cache notes."""
    midi_file_count = len(list(dataset_dir.glob("*.mid")))
    notes = extract_notes(dataset_dir=dataset_dir, force=force)
    return DatasetStats(
        midi_file_count=midi_file_count,
        note_count=len(notes),
        vocab_size=len(set(notes)),
    )
