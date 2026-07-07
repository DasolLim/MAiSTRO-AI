"""Render a MIDI file to a playable WAV using a soundfont, with light post-processing.

Merged from the near-duplicate midi_to_audio_c.py ("collaboration" mode: audio
before 15s plays at the original tempo) and midi_to_audio_g.py ("generation"
mode: the tempo slow-down applies from the very start). The only difference
between the two was the `split_time` constant.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from pydub import AudioSegment
from pypianoroll import read, to_pretty_midi

from . import config

SAMPLE_RATE = 44100
TIME_SCALING_FACTOR = 1.225

SPLIT_TIME_BY_MODE = {
    "collaboration": 15.0,
    "generation": 0.0,
}


def render_midi_to_wav(midi_path: Path, mode: str = "generation") -> Path:
    """Render midi_path to a .wav file next to it and return the wav path."""
    if mode not in SPLIT_TIME_BY_MODE:
        raise ValueError(f"mode must be one of {list(SPLIT_TIME_BY_MODE)}")
    split_time = SPLIT_TIME_BY_MODE[mode]

    midi_path = Path(midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    multitrack = read(str(midi_path))
    pm = to_pretty_midi(multitrack)

    original_tempo = pm.estimate_tempo() if pm.get_tempo_changes()[1].size > 0 else 120.0
    adjusted_tempo = original_tempo / TIME_SCALING_FACTOR

    new_pm = pretty_midi.PrettyMIDI()
    original_tempo_event = pretty_midi.ControlChange(
        number=51, value=int(60_000_000 / original_tempo), time=0
    )
    adjusted_tempo_event = pretty_midi.ControlChange(
        number=51, value=int(60_000_000 / adjusted_tempo), time=split_time
    )

    for instrument in pm.instruments:
        new_instrument = pretty_midi.Instrument(program=0)
        if not new_pm.instruments:
            new_instrument.control_changes.extend([original_tempo_event, adjusted_tempo_event])

        for note in instrument.notes:
            if note.end <= split_time:
                new_start, new_end = note.start, note.end
            elif note.start >= split_time:
                new_start = split_time + (note.start - split_time) * TIME_SCALING_FACTOR
                new_end = split_time + (note.end - split_time) * TIME_SCALING_FACTOR
            else:
                new_start = note.start
                new_end = split_time + (note.end - split_time) * TIME_SCALING_FACTOR

            time_variation = np.random.uniform(-0.01, 0.01)
            new_start = max(0.0, new_start + time_variation)
            new_end = max(new_start + 0.05, new_end + time_variation)

            new_instrument.notes.append(
                pretty_midi.Note(velocity=90, pitch=note.pitch, start=new_start, end=new_end)
            )

        new_pm.instruments.append(new_instrument)

    wave = new_pm.fluidsynth(sf2_path=str(config.SOUNDFONT_PATH), fs=SAMPLE_RATE)

    unprocessed_path = midi_path.with_suffix(".unprocessed.wav")
    sf.write(str(unprocessed_path), wave, SAMPLE_RATE)

    segment = AudioSegment.from_wav(str(unprocessed_path))
    filtered = segment.low_pass_filter(500)
    filtered = filtered.overlay(filtered - 6, position=50)
    filtered = filtered.normalize()

    wav_path = midi_path.with_suffix(".wav")
    filtered.export(str(wav_path), format="wav")
    unprocessed_path.unlink(missing_ok=True)

    return wav_path
