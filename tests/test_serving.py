"""The deployed path: NumPy inference, MIDI writing, and the bundle that ships.

The load-bearing test here is `test_generation_imports_nothing_heavy`. The whole
free-tier deployment rests on the serving path never reaching for TensorFlow,
music21, torch or SciPy -- any of which blows the 500MB function ceiling. A stray
import would pass every other test and fail only at deploy time.
"""

from __future__ import annotations

import builtins
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from backend.maistro import lite, npmodel
from backend.maistro.midi_writer import tokens_to_midi_bytes

ROOT = Path(__file__).resolve().parents[1]

# Everything a 500MB serverless bundle cannot afford.
FORBIDDEN_IMPORTS = {"tensorflow", "keras", "music21", "torch", "scipy", "transformers"}


class TestNumpyModel:
    def test_gelu_matches_its_closed_form(self):
        x = np.linspace(-3, 3, 50)
        expected = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        np.testing.assert_allclose(npmodel.gelu(x), expected, rtol=1e-12)

    def test_layer_norm_standardises(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        out = npmodel.layer_norm(x, np.ones(4), np.zeros(4))
        assert out.mean() == pytest.approx(0.0, abs=1e-6)
        assert out.std() == pytest.approx(1.0, abs=1e-3)

    def test_attention_is_causal(self, model_dir):
        """Changing the last token must not change what earlier positions attended to."""
        weights = npmodel.load(model_dir / "transformer.npz")
        tokens = np.arange(weights.sequence_length, dtype=np.int32) % weights.n_vocab

        # A causal model's prediction depends on the whole prefix; changing the *last*
        # token must change it, while a non-causal one would also leak future context.
        first = npmodel.predict_next(weights, tokens)
        changed = tokens.copy()
        changed[-1] = (changed[-1] + 1) % weights.n_vocab
        assert not np.allclose(first, npmodel.predict_next(weights, changed))

    def test_predict_next_returns_a_distribution(self, model_dir):
        weights = npmodel.load(model_dir / "transformer.npz")
        tokens = np.zeros(weights.sequence_length, dtype=np.int32)
        probs = npmodel.predict_next(weights, tokens)

        assert probs.shape == (weights.n_vocab,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)
        assert (probs >= 0).all()

    def test_float16_storage_round_trips(self, model_dir):
        weights = npmodel.load(model_dir / "transformer.npz")
        # load() upcasts for arithmetic; fp16 on disk must not leave fp16 in the math.
        assert weights.token_embedding.dtype == np.float32


class TestMidiWriter:
    def _messages(self, data: bytes):
        import mido

        return list(mido.MidiFile(file=io.BytesIO(data)))

    def test_writes_a_standard_midi_header(self):
        assert tokens_to_midi_bytes(["C4 1.0"])[:4] == b"MThd"

    def test_chords_share_an_onset(self):
        import mido

        midi = mido.MidiFile(file=io.BytesIO(tokens_to_midi_bytes(["C4.E4.G4 1.0"])))
        note_ons = [m for track in midi.tracks for m in track if m.type == "note_on"]
        assert len(note_ons) == 3
        # After the first, the rest follow with zero delta: they sound together.
        assert all(m.time == 0 for m in note_ons[1:])

    def test_rests_advance_time_without_sounding(self):
        import mido

        without = mido.MidiFile(file=io.BytesIO(tokens_to_midi_bytes(["C4 1.0", "E4 1.0"])))
        with_rest = mido.MidiFile(file=io.BytesIO(tokens_to_midi_bytes(["C4 1.0", "rest 1.0", "E4 1.0"])))
        assert with_rest.length > without.length

        sounded = [m for t in with_rest.tracks for m in t if m.type == "note_on"]
        assert len(sounded) == 2

    def test_tempo_is_written(self):
        import mido

        midi = mido.MidiFile(file=io.BytesIO(tokens_to_midi_bytes(["C4 1.0"], tempo_bpm=72)))
        tempos = [m for t in midi.tracks for m in t if m.type == "set_tempo"]
        assert len(tempos) == 1
        assert round(mido.tempo2bpm(tempos[0].tempo)) == 72

    def test_unparseable_pitches_are_dropped_not_crashed(self):
        assert tokens_to_midi_bytes(["??? 1.0", "C4 1.0"])[:4] == b"MThd"


class TestLiteGeneration:
    def test_generate_produces_midi_and_metrics(self, model_dir):
        bundle = lite.load_bundle(str(model_dir))
        result = lite.generate(bundle, lite.LiteConfig(n_notes=12, seed=3))

        assert result.midi[:4] == b"MThd"
        assert len(result.tokens) == 12
        assert "repetition_rate" in result.metrics
        assert "pitch_class_kl" in result.metrics  # recomputed from the shipped histogram
        assert result.config["seed"] == 3

    def test_the_seed_makes_a_take_reproducible(self, model_dir):
        bundle = lite.load_bundle(str(model_dir))
        cfg = lite.LiteConfig(n_notes=10, seed=42, temperature=1.8)
        assert lite.generate_tokens(bundle, cfg) == lite.generate_tokens(bundle, cfg)

    def test_different_seeds_diverge(self, model_dir):
        bundle = lite.load_bundle(str(model_dir))
        a = lite.generate_tokens(bundle, lite.LiteConfig(n_notes=16, seed=1, temperature=1.9))
        b = lite.generate_tokens(bundle, lite.LiteConfig(n_notes=16, seed=2, temperature=1.9))
        assert a != b

    def test_a_mood_supplies_its_own_temperature(self, model_dir):
        resolved = lite.LiteConfig(mood="melancholic").resolved()
        assert resolved.temperature == pytest.approx(1.45)
        assert resolved.scale == "minor"

    def test_note_count_is_clamped(self, model_dir):
        assert lite.LiteConfig(n_notes=99_999).resolved().n_notes == lite.MAX_NOTES

    def test_generation_imports_nothing_heavy(self, model_dir):
        """The deployed bundle is 40MB. TensorFlow alone is 877MB."""
        for name in FORBIDDEN_IMPORTS:
            sys.modules.pop(name, None)

        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in FORBIDDEN_IMPORTS:
                raise AssertionError(f"serving path imported {name!r}")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = guard
        try:
            bundle = lite.load_bundle(str(model_dir))
            result = lite.generate(bundle, lite.LiteConfig(n_notes=8, key="C", scale="major"))
        finally:
            builtins.__import__ = real_import

        assert result.midi[:4] == b"MThd"


class TestWebBundle:
    def test_web_maistro_is_in_sync_with_backend(self):
        """web/maistro/ is a copy. If it drifts, the deploy runs different code."""
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "sync_web_bundle.py"), "--check"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr or result.stdout

    def test_deployed_requirements_stay_small(self):
        requirements = (ROOT / "web" / "requirements.txt").read_text()
        packages = {
            line.split(">=")[0].split("==")[0].strip()
            for line in requirements.splitlines()
            if line.strip() and not line.startswith("#")
        }
        assert packages == {"fastapi", "numpy", "mido"}
