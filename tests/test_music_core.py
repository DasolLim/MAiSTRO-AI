"""Theory, sampling, conditioning and metrics: the pure core the decoder is built on."""

from __future__ import annotations

import numpy as np
import pytest

from backend.maistro import metrics, theory
from backend.maistro.conditioning import Conditioning, build_logit_bias, vocabulary_features
from backend.maistro.sampling import apply_top_k, apply_top_p, sample_token, softmax


class TestTheory:
    @pytest.mark.parametrize(
        "name,pitch_class,midi",
        [
            ("C4", 0, 60),
            ("C#4", 1, 61),
            ("B-3", 10, 58),  # music21 spells a flat as "-"
            ("A0", 9, 21),
            ("G#7", 8, 104),
            ("C", 0, None),  # no octave means no MIDI number
        ],
    )
    def test_note_names(self, name, pitch_class, midi):
        assert theory.pitch_class(name) == pitch_class
        assert theory.midi_number(name) == midi

    def test_scales(self):
        assert sorted(theory.scale_pitch_classes("C", "major")) == [0, 2, 4, 5, 7, 9, 11]
        assert sorted(theory.scale_pitch_classes("A", "minor")) == [0, 2, 4, 5, 7, 9, 11]
        assert len(theory.scale_pitch_classes("D", "chromatic")) == 12

    def test_parse_chord_rest_and_fraction(self):
        chord = theory.parse_token("C4.E4.G4 1.0")
        assert chord.pitch_classes == frozenset({0, 4, 7})
        assert chord.mean_midi == pytest.approx(64.0, abs=0.5)

        rest = theory.parse_token("rest 0.5")
        assert rest.is_rest and rest.duration == 0.5 and not rest.pitch_classes

        triplet = theory.parse_token("F#5 1/3")
        assert triplet.duration == pytest.approx(1 / 3)

    def test_every_mood_preset_is_usable(self):
        for mood in theory.MOODS.values():
            assert mood.suggested_scale in theory.SCALES
            # Calibrated to this model's confidence: below ~1.3 sampling is greedy.
            assert 1.3 <= mood.temperature <= 2.1
            assert 0.9 <= mood.top_p <= 1.0


class TestSampling:
    def test_temperature_zero_is_greedy(self):
        probs = np.array([0.1, 0.7, 0.2])
        assert all(sample_token(probs, temperature=0.0) == 1 for _ in range(5))

    def test_top_k_one_is_deterministic(self):
        probs = np.array([0.3, 0.5, 0.2])
        rng = np.random.default_rng(0)
        picks = {sample_token(probs, temperature=2.0, top_k=1, rng=rng) for _ in range(20)}
        assert picks == {1}

    def test_sampling_explores_when_temperature_is_high(self):
        probs = np.array([0.7, 0.2, 0.09, 0.01])
        rng = np.random.default_rng(1)
        picks = {sample_token(probs, temperature=2.0, rng=rng) for _ in range(50)}
        assert len(picks) > 1

    def test_logit_bias_can_veto_a_token(self):
        probs = np.array([0.9, 0.1])
        bias = np.array([-20.0, 0.0])  # crush the favourite
        assert sample_token(probs, temperature=1.0, logit_bias=bias) == 1

    def test_top_p_keeps_at_least_one_token(self):
        # One token holds more mass than the threshold; the nucleus must not be empty.
        logits = np.log(np.array([0.99, 0.005, 0.005]))
        kept = np.isfinite(apply_top_p(logits, 0.5))
        assert kept.sum() == 1

    def test_top_k_masks_the_tail(self):
        logits = np.array([3.0, 2.0, 1.0, 0.0])
        assert np.isfinite(apply_top_k(logits, 2)).sum() == 2

    def test_softmax_normalises(self):
        assert softmax(np.array([1.0, 2.0, 3.0])).sum() == pytest.approx(1.0)


class TestConditioning:
    def test_out_of_key_tokens_are_penalised(self):
        pitchnames = ["C4 1.0", "C#4 1.0", "E4 1.0"]  # C# is outside C major
        features = vocabulary_features(pitchnames)
        bias = build_logit_bias(features, Conditioning(key="C", scale="major"))

        assert bias is not None
        assert bias[1] < bias[0]
        assert bias[0] == pytest.approx(bias[2])  # both in key

    def test_no_conditioning_means_no_bias(self):
        features = vocabulary_features(["C4 1.0"])
        assert build_logit_bias(features, Conditioning()) is None

    def test_rest_bias_moves_rests_only(self):
        features = vocabulary_features(["C4 1.0", "rest 1.0"])
        more_space = build_logit_bias(features, Conditioning(rest_bias=1.0))
        assert more_space[1] > more_space[0]


class TestMetrics:
    def test_repetition_rate(self):
        assert metrics.repetition_rate(["a", "b", "c", "d", "e"], n=4) == 0.0
        looped = ["a", "b", "c", "d"] * 4
        assert metrics.repetition_rate(looped, n=4) > 0.5

    def test_kl_divergence_is_zero_for_identical_distributions(self):
        p = np.array([0.25, 0.25, 0.5])
        assert metrics.kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)
        assert metrics.kl_divergence(p, np.array([0.5, 0.25, 0.25])) > 0

    def test_pitch_class_histogram_sums_to_one(self):
        histogram = metrics.pitch_class_histogram(["C4 1.0", "E4 1.0", "G4 1.0"])
        assert histogram.sum() == pytest.approx(1.0)
        assert histogram[0] == pytest.approx(1 / 3)

    def test_evaluate_reports_the_expected_fields(self):
        scores = metrics.evaluate(["C4 1.0", "E4 0.5", "rest 0.5"])
        for field in ("note_count", "repetition_rate", "rest_fraction", "distinct_pitch_classes"):
            assert field in scores
        assert scores["rest_fraction"] == pytest.approx(1 / 3)
