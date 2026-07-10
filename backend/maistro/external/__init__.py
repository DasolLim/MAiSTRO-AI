"""Adapters for third-party generative-audio models.

MAiSTRO's own model is symbolic: it predicts notes, and a soundfont turns those
notes into sound. That is the right tool for classical piano and the wrong tool
for anything that lives in timbre rather than notation -- a synth pad, a drum
break, a guitar tone. The adapters here reach for models that generate audio
waveforms directly, so the app can cover genres the note vocabulary cannot
express, while keeping every dependency optional.
"""
