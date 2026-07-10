"""Generate a new MIDI composition from a trained model.

Decoding is configurable end to end: pick an architecture, a temperature, a
nucleus, and a key/mood to steer toward. The two decisions that matter most:

* Sampling replaced `argmax`. See sampling.py for why the old decoder looped.
* Conditioning is applied as a logit bias, not a model input. See conditioning.py.

Every generation writes a `.json` sidecar next to its `.mid` recording the exact
config and RNG seed used, so any result in the library can be reproduced.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from music21 import chord, instrument, note, stream, tempo

from . import architectures, config, metrics
from .conditioning import Conditioning, build_logit_bias, choose_seed_index, vocabulary_features
from .sampling import sample_token
from .theory import MOODS

DEFAULT_NOTES_TO_GENERATE = 300

# Calibrated against the trained model's measured output entropy, not copied from
# text-LM defaults — see the table in sampling.py. T=1.0 here would be greedy.
DEFAULT_TEMPERATURE = 1.5
DEFAULT_TOP_P = 0.98
DEFAULT_TOP_K = 40


@dataclass
class GenerationConfig:
    """Everything that determines the output, including the RNG seed."""

    arch: str = architectures.DEFAULT_ARCHITECTURE
    n_notes: int = DEFAULT_NOTES_TO_GENERATE

    # None means "inherit from the mood preset", falling back to the tuned default.
    temperature: float | None = None
    top_k: int = DEFAULT_TOP_K
    top_p: float | None = None

    key: str | None = None
    scale: str = "chromatic"
    mood: str | None = None
    tempo_bpm: int = 96

    seed: int | None = None

    def resolved(self) -> "GenerationConfig":
        """Fill temperature/top_p/scale from the mood preset where the caller left them blank."""
        mood = MOODS.get(self.mood or "")
        return GenerationConfig(
            arch=self.arch,
            n_notes=self.n_notes,
            temperature=self.temperature
            if self.temperature is not None
            else (mood.temperature if mood else DEFAULT_TEMPERATURE),
            top_k=self.top_k,
            top_p=self.top_p if self.top_p is not None else (mood.top_p if mood else DEFAULT_TOP_P),
            key=self.key,
            scale=self.scale if self.scale != "chromatic" else (mood.suggested_scale if mood else "chromatic"),
            mood=self.mood,
            tempo_bpm=self.tempo_bpm,
            seed=self.seed,
        )

    def to_conditioning(self) -> Conditioning:
        mood = MOODS.get(self.mood or "")
        return Conditioning(
            key=self.key,
            scale=self.scale,
            mood=self.mood,
            register_center=mood.register_center if mood else None,
            register_width=mood.register_width if mood else 12.0,
            duration_pref=mood.duration_pref if mood else 0.0,
            rest_bias=mood.rest_bias if mood else 0.0,
        )


@dataclass
class GenerationResult:
    midi_path: Path
    tokens: list[str] = field(repr=False)
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def load_vocabulary() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(notes, pitchnames) from the prepared dataset. Cached; the notes file is static."""
    if not config.NOTES_FILE.exists():
        raise FileNotFoundError(
            f"No prepared notes at {config.NOTES_FILE}. Run the dataset step first."
        )
    with open(config.NOTES_FILE, "rb") as fh:
        notes = pickle.load(fh)
    return tuple(notes), tuple(sorted(set(notes)))


@lru_cache(maxsize=len(architectures.ARCHITECTURES))
def _load_model(arch: str, n_vocab: int, weights_key: str):
    """Cached model load. `weights_key` busts the cache when a new checkpoint lands."""
    return architectures.load_trained_network(arch, n_vocab, Path(weights_key))


def load_model(arch: str, n_vocab: int):
    weights_path = config.latest_weights_file(arch)
    return _load_model(arch, n_vocab, str(weights_path))


def build_windows(notes: tuple[str, ...], pitchnames: tuple[str, ...]) -> list[list[int]]:
    """All sliding windows of token ids that generation can be seeded from."""
    note_to_int = {n: i for i, n in enumerate(pitchnames)}
    sequence_length = config.SEQUENCE_LENGTH
    return [
        [note_to_int[token] for token in notes[i : i + sequence_length]]
        for i in range(len(notes) - sequence_length)
    ]


def generate_tokens(cfg: GenerationConfig) -> list[str]:
    """Autoregressively sample `cfg.n_notes` vocabulary tokens."""
    cfg = cfg.resolved()
    notes, pitchnames = load_vocabulary()
    n_vocab = len(pitchnames)

    features = vocabulary_features(list(pitchnames))
    conditioning = cfg.to_conditioning()
    logit_bias = build_logit_bias(features, conditioning)

    rng = np.random.default_rng(cfg.seed)
    windows = build_windows(notes, pitchnames)
    pattern = list(windows[choose_seed_index(windows, features, conditioning, rng)])

    model = load_model(cfg.arch, n_vocab)
    encoding = architectures.get(cfg.arch).encoding

    generated: list[str] = []
    for _ in range(cfg.n_notes):
        model_input = architectures.encode_windows([pattern], n_vocab, encoding)
        # __call__ skips the tf.data/callback machinery predict() sets up per call,
        # which dominates runtime when the batch is a single 100-step window.
        probs = np.asarray(model(model_input, training=False))[0]

        index = sample_token(
            probs,
            temperature=cfg.temperature or 0.0,
            top_k=cfg.top_k,
            top_p=cfg.top_p or 0.0,
            logit_bias=logit_bias,
            rng=rng,
        )
        generated.append(pitchnames[index])
        pattern = pattern[1:] + [index]

    return generated


def tokens_to_midi(tokens: list[str], filepath: Path, tempo_bpm: int = 96) -> None:
    """Convert generated note/chord/rest tokens into a MIDI file."""
    midi_stream = stream.Stream()
    midi_stream.insert(0.0, tempo.MetronomeMark(number=tempo_bpm))

    offset = 0.0
    for token in tokens:
        name, _, duration_str = token.partition(" ")
        duration = _to_float(duration_str)

        if ("." in name) or name.isdigit():
            pitches = [
                note.Note(int(p)) if p.isdigit() else note.Note(p) for p in name.split(".")
            ]
            for pitch_note in pitches:
                pitch_note.storedInstrument = instrument.Piano()
            element = chord.Chord(pitches)
        elif "rest" in name:
            element = note.Rest()
            duration = min(duration, 4.0)  # a longer rest is a bug in the token, not a phrase
        else:
            element = note.Note(name)
            element.storedInstrument = instrument.Piano()

        # The original writer built the stream with Stream(list), whose append()
        # recomputes every offset from the running duration -- and it never set a
        # duration, so every token came out a quarter note regardless of what the
        # model predicted. Durations are honoured and offsets inserted explicitly.
        if duration > 0:
            element.duration.quarterLength = duration
        midi_stream.insert(offset, element)
        offset += duration

    midi_stream.write("midi", fp=str(filepath))
    print(f"Generated MIDI saved as {filepath}")


def _to_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        pass
    if "/" in text:
        num, _, denom = text.partition("/")
        whole = 0.0
        if " " in num:  # mixed number, e.g. "1 1/3"
            leading, _, num = num.partition(" ")
            whole = float(leading)
        try:
            fraction = float(num) / float(denom)
        except (ValueError, ZeroDivisionError):
            return whole
        return whole - fraction if whole < 0 else whole + fraction
    return 0.0


def generate(
    output_filename: str = "generated_output.mid",
    cfg: GenerationConfig | None = None,
    write_sidecar: bool = True,
) -> GenerationResult:
    """Generate one composition, write it (plus a config sidecar), and score it.

    `write_sidecar=False` suppresses the `.json` companion. The arena needs that:
    the sidecar names the architecture, and every file under output/ is reachable
    through GET /audio/file, so writing one next to a blinded clip would let a
    listener unblind the test by opening a URL.
    """
    cfg = (cfg or GenerationConfig()).resolved()
    if cfg.seed is None:
        # Materialise the seed so the sidecar can reproduce this exact take.
        cfg.seed = int(np.random.SeedSequence().entropy % (2**32))

    tokens = generate_tokens(cfg)

    output_path = config.OUTPUT_DIR / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokens_to_midi(tokens, output_path, tempo_bpm=cfg.tempo_bpm)

    scores = metrics.evaluate(tokens)
    if write_sidecar:
        sidecar = {"config": asdict(cfg), "metrics": scores}
        output_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))

    return GenerationResult(
        midi_path=output_path, tokens=tokens, metrics=scores, config=asdict(cfg)
    )
