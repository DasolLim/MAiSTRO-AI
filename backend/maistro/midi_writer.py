"""Write note tokens to MIDI bytes using mido, with no music21.

music21 is a superb analysis library and 111MB on disk. Writing a single-track
piano roll needs none of that: a tempo meta message, then note_on/note_off pairs
at the right ticks. mido is under 1MB and this keeps the deployed bundle small
enough for a serverless function.

The output is byte-compatible with what generate.tokens_to_midi produces via
music21 -- same pitches, same onsets, same durations -- so anything that plays one
plays the other.
"""

from __future__ import annotations

import io

from .theory import midi_number, parse_token

TICKS_PER_BEAT = 480
DEFAULT_VELOCITY = 90
# A rest longer than this is a bug in the token, not a phrase. Matches generate.py.
MAX_REST_QUARTERS = 4.0
MIN_DURATION_QUARTERS = 0.0625  # a 64th note; anything shorter is inaudible


def tokens_to_midi_bytes(tokens: list[str], tempo_bpm: int = 96) -> bytes:
    """Render note/chord/rest tokens to the bytes of a standard MIDI file."""
    import mido

    midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    track.append(mido.Message("program_change", program=0, time=0))  # acoustic grand

    # Collect absolute-tick events first; mido wants delta times, and chords mean
    # several notes share an onset.
    events: list[tuple[int, int, int, int]] = []  # (tick, is_note_on, pitch, order)
    offset_quarters = 0.0

    for token in tokens:
        features = parse_token(token)

        if features.is_rest:
            offset_quarters += min(features.duration, MAX_REST_QUARTERS)
            continue

        duration = max(features.duration, MIN_DURATION_QUARTERS)
        name = token.split(" ", 1)[0]

        for part in name.split("."):
            pitch = int(part) + 60 if part.isdigit() else midi_number(part)
            if pitch is None or not 0 <= pitch <= 127:
                continue
            start = int(round(offset_quarters * TICKS_PER_BEAT))
            end = int(round((offset_quarters + duration) * TICKS_PER_BEAT))
            events.append((start, 1, pitch, 0))
            events.append((max(end, start + 1), 0, pitch, 0))

        offset_quarters += duration

    # note_off before note_on at the same tick, so a repeated pitch retriggers
    # rather than being cut short by its predecessor's release.
    events.sort(key=lambda e: (e[0], e[1]))

    previous_tick = 0
    for tick, is_note_on, pitch, _ in events:
        track.append(
            mido.Message(
                "note_on" if is_note_on else "note_off",
                note=pitch,
                velocity=DEFAULT_VELOCITY if is_note_on else 0,
                time=tick - previous_tick,
            )
        )
        previous_tick = tick

    buffer = io.BytesIO()
    midi.save(file=buffer)
    return buffer.getvalue()
