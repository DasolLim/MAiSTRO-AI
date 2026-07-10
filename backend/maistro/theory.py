"""Music-theory primitives for steering and evaluating generated note sequences.

Vocabulary tokens look like `"C#4 0.5"` (note), `"C4.E4.G4 1.0"` (chord) or
`"rest 0.5"` -- see dataset.extract_notes. Everything here parses that format
without pulling in music21, which is far too slow to call once per vocabulary
entry per generation request.
"""

from __future__ import annotations

from dataclasses import dataclass

# Semitone offset of each natural note from C.
_NATURAL_PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# music21 writes flats as "-" (B-4); accept "b" too since users type that.
_ACCIDENTALS = {"#": 1, "-": -1, "b": -1}

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Scale degrees as semitone offsets from the tonic.
SCALES: dict[str, tuple[int, ...]] = {
    "chromatic": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
}


@dataclass(frozen=True)
class Mood:
    """A named preset over the sampling + conditioning dials.

    These are inference-time steering presets, not learned emotion labels: each
    one is a point in (temperature, register, density) space that reliably
    produces the character it is named after.

    The temperatures look high because they are calibrated against this model's
    measured output distribution rather than copied from text-LM defaults. The
    trained network is very confident (mean max probability ~0.95, entropy ~0.21
    nats), so anything below T≈1.3 samples the argmax token nearly every step and
    is indistinguishable from greedy decoding. See sampling.py.
    """

    key: str
    label: str
    description: str
    temperature: float
    top_p: float
    suggested_scale: str
    register_center: int | None  # MIDI note number the melody is pulled toward
    register_width: float  # semitones; larger = looser pull
    duration_pref: float  # >0 favours long notes, <0 favours short ones
    rest_bias: float  # >0 favours rests (space), <0 fills the bar


MOODS: dict[str, Mood] = {
    "serene": Mood(
        key="serene",
        label="Serene",
        description="Sparse, unhurried, mid-register. Nocturne territory.",
        temperature=1.35,
        top_p=0.97,
        suggested_scale="major",
        register_center=64,
        register_width=10.0,
        duration_pref=0.9,
        rest_bias=0.5,
    ),
    "melancholic": Mood(
        key="melancholic",
        label="Melancholic",
        description="Low register, minor colour, long lines.",
        temperature=1.45,
        top_p=0.98,
        suggested_scale="minor",
        register_center=55,
        register_width=11.0,
        duration_pref=0.6,
        rest_bias=0.2,
    ),
    "triumphant": Mood(
        key="triumphant",
        label="Triumphant",
        description="High register, dense, major. Ceremonial.",
        temperature=1.6,
        top_p=0.98,
        suggested_scale="major",
        register_center=74,
        register_width=12.0,
        duration_pref=-0.5,
        rest_bias=-0.8,
    ),
    "turbulent": Mood(
        key="turbulent",
        label="Turbulent",
        description="Fast, chromatic, restless. Leaves the key when it wants to.",
        temperature=1.95,
        top_p=0.99,
        suggested_scale="harmonic_minor",
        register_center=None,
        register_width=18.0,
        duration_pref=-1.0,
        rest_bias=-0.6,
    ),
    "playful": Mood(
        key="playful",
        label="Playful",
        description="Bright, bouncing, short notes with room to breathe.",
        temperature=1.75,
        top_p=0.98,
        suggested_scale="lydian",
        register_center=70,
        register_width=13.0,
        duration_pref=-0.7,
        rest_bias=0.3,
    ),
}


def _split_note(note_name: str) -> tuple[int, str] | None:
    """Split `C#4` into (semitones-from-C including accidentals, octave digits).

    Accidentals are consumed left-to-right. Note that music21 spells a flat as
    "-", so "C-1" is C-flat in octave 1, not C in octave -1; this follows that
    convention because it is the one dataset.extract_notes wrote the tokens in.
    """
    if not note_name:
        return None
    letter = note_name[0].upper()
    if letter not in _NATURAL_PITCH_CLASSES:
        return None

    value = _NATURAL_PITCH_CLASSES[letter]
    index = 1
    while index < len(note_name) and note_name[index] in _ACCIDENTALS:
        value += _ACCIDENTALS[note_name[index]]
        index += 1

    return value, note_name[index:]


def pitch_class(note_name: str) -> int | None:
    """Pitch class 0-11 for a note name like `C#4` / `B-3`, or None if unparseable."""
    split = _split_note(note_name)
    return None if split is None else split[0] % 12


def midi_number(note_name: str) -> int | None:
    """MIDI note number for a name like `C#4`, or None if it carries no octave."""
    split = _split_note(note_name)
    if split is None:
        return None
    semitones, octave_text = split
    if not octave_text.isdigit():
        return None
    return (int(octave_text) + 1) * 12 + semitones


def scale_pitch_classes(tonic: str, scale: str) -> frozenset[int]:
    """The set of pitch classes in `scale` rooted at `tonic` (e.g. "D", "minor")."""
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; expected one of {sorted(SCALES)}")
    root = pitch_class(tonic)
    if root is None:
        raise ValueError(f"unknown tonic {tonic!r}")
    return frozenset((root + step) % 12 for step in SCALES[scale])


@dataclass(frozen=True)
class TokenFeatures:
    """Everything conditioning and metrics need to know about one vocabulary entry."""

    token: str
    is_rest: bool
    pitch_classes: frozenset[int]
    mean_midi: float | None
    duration: float


def _parse_duration(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        pass
    if "/" in text:
        num, _, denom = text.partition("/")
        try:
            return float(num) / float(denom)
        except (ValueError, ZeroDivisionError):
            return 0.0
    return 0.0


def parse_token(token: str) -> TokenFeatures:
    """Decompose a `"<names> <duration>"` vocabulary token into its features."""
    name, _, duration_str = token.partition(" ")
    duration = _parse_duration(duration_str)

    if "rest" in name:
        return TokenFeatures(token, True, frozenset(), None, duration)

    names = name.split(".")
    classes = {pc for pc in (pitch_class(n) for n in names) if pc is not None}
    midis = [m for m in (midi_number(n) for n in names) if m is not None]

    return TokenFeatures(
        token=token,
        is_rest=False,
        pitch_classes=frozenset(classes),
        mean_midi=sum(midis) / len(midis) if midis else None,
        duration=duration,
    )


def parse_vocabulary(pitchnames: list[str]) -> list[TokenFeatures]:
    """Parse every vocabulary token once, in vocabulary order."""
    return [parse_token(token) for token in pitchnames]
