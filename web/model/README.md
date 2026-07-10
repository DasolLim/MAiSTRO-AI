# Deployed model

Two files belong here, produced by `jupyter_notebooks/train_transformer_colab.ipynb`:

| File | Size | Contents |
|---|---:|---|
| `transformer.npz` | ~7 MB | Trained transformer weights, float16 |
| `vocabulary.npz` | ~0.2 MB | 3,388 note tokens, the seed corpus, its pitch-class histogram |

They are the *only* model artefacts the deployed API reads — no Keras checkpoint, no
TensorFlow. Until they exist, `POST /generate/sync` returns a 503 saying so.

Commit them: 7 MB is well within what git handles comfortably, and the deploy needs
them present in the Root Directory.
