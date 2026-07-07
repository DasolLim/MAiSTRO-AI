"""Check whether a generated MIDI's note distribution is suspiciously close to
any single track in the training dataset (a sign of overfitting/memorization).

similarity ~1.0  -> near-exact copy of that dataset track
similarity > 0.8 -> very similar, model may be memorizing
similarity 0.5-0.7 -> good balance of structure and variation
similarity < 0.4 -> very different, model is generating novel material
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
from music21 import chord, converter, note
from sklearn.metrics.pairwise import cosine_similarity

from . import config


def extract_note_names(midi_path: Path) -> list[str]:
    midi = converter.parse(str(midi_path))
    names = []
    for element in midi.flat.notes:
        if isinstance(element, note.Note):
            names.append(element.nameWithOctave)
        elif isinstance(element, chord.Chord):
            names.append(".".join(p.nameWithOctave for p in element.pitches))
    return names


def note_frequencies(midi_path: Path) -> collections.Counter:
    return collections.Counter(extract_note_names(midi_path))


def cosine_similarity_between(counts1: collections.Counter, counts2: collections.Counter) -> float:
    unique_notes = sorted(set(counts1) | set(counts2))
    vector1 = np.array([counts1.get(n, 0) for n in unique_notes])
    vector2 = np.array([counts2.get(n, 0) for n in unique_notes])

    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        return 0.0

    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)
    return float(cosine_similarity([vector1], [vector2])[0][0])


def check_overfit(generated_midi: Path, dataset_dir: Path = config.DATASET_DIR) -> list[tuple[str, float]]:
    """Compare a generated MIDI against every dataset track, most similar first."""
    generated_counts = note_frequencies(generated_midi)

    results = []
    for dataset_file in sorted(dataset_dir.glob("*.mid")):
        similarity = cosine_similarity_between(note_frequencies(dataset_file), generated_counts)
        results.append((str(dataset_file), similarity))

    results.sort(key=lambda item: item[1], reverse=True)
    return results
