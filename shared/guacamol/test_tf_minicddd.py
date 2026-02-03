"""Test miniCDDD encoder directly with TensorFlow/Keras."""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import json
from pathlib import Path

# Import TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Path to models
MODEL_DIR = Path(__file__).parent / "minicddd_models"
LOOKUP_PATH = MODEL_DIR / "lookup_table.json"

# Load vocabulary
with open(LOOKUP_PATH) as f:
    VOCAB = json.load(f)
VOCAB_SIZE = len(VOCAB)
print(f"Vocab size: {VOCAB_SIZE}")


def tokenize_smiles(smiles: str, max_length: int = 73) -> np.ndarray:
    """Tokenize SMILES string."""
    import re
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    atoms = re.findall(pattern, smiles)

    # SOS + tokens + EOS + padding
    tokens = [VOCAB.get("<SOS>", 44)]
    for atom in atoms:
        if atom in VOCAB:
            tokens.append(VOCAB[atom])
    tokens.append(VOCAB.get("<EOS>", 45))

    # Pad
    pad_id = VOCAB.get("<PAD>", 43)
    while len(tokens) < max_length:
        tokens.append(pad_id)

    return np.array(tokens[:max_length], dtype=np.int32)


class OneHotLayer(layers.Layer):
    """One-hot encoding layer compatible with Keras 3."""
    def __init__(self, depth: int, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth

    def call(self, inputs):
        return tf.one_hot(inputs, self.depth)

    def get_config(self):
        config = super().get_config()
        config["depth"] = self.depth
        return config


def build_encoder(vocab_size: int = 46, latent_dim: int = 512):
    """Build miniCDDD encoder model."""
    # Input layer
    encoder_inputs = keras.Input(shape=(None,), dtype=tf.int32, name="encoder_inputs")

    # One-hot encoding using custom layer
    x = OneHotLayer(depth=vocab_size)(encoder_inputs)

    # GRU layers
    gru1 = layers.GRU(512, return_sequences=True, return_state=True, name="gru")
    gru2 = layers.GRU(1024, return_sequences=True, return_state=True, name="gru_1")
    gru3 = layers.GRU(2048, return_sequences=True, return_state=True, name="gru_2")

    out1, state1 = gru1(x)
    out2, state2 = gru2(out1)
    _, state3 = gru3(out2)

    # Concatenate states
    concat = layers.Concatenate()([state1, state2, state3])

    # Dense to latent
    latent = layers.Dense(latent_dim, activation="tanh", name="dense")(concat)

    model = keras.Model(encoder_inputs, latent, name="encoder")
    return model


def main():
    print("Building encoder model...")
    encoder = build_encoder(VOCAB_SIZE, 512)
    encoder.summary()

    # Load weights from H5 file
    encoder_path = MODEL_DIR / "encoder.h5"
    print(f"\nLoading weights from {encoder_path}...")

    try:
        encoder.load_weights(str(encoder_path))
        print("Weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # Test encoding
    test_smiles = ["CCO", "CCC", "c1ccccc1", "CC(=O)O"]
    print(f"\nTest SMILES: {test_smiles}")

    # Tokenize
    tokens = np.stack([tokenize_smiles(s) for s in test_smiles])
    print(f"Token shapes: {tokens.shape}")
    print(f"First tokens (CCO): {tokens[0][:10]}")

    # Encode
    embeddings = encoder.predict(tokens, verbose=0)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embedding norms: {[f'{np.linalg.norm(e):.4f}' for e in embeddings]}")

    # Check cosine similarities
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print(f"\nCosine similarities:")
    print(f"  CCO vs CCC: {cosine_sim(embeddings[0], embeddings[1]):.4f}")
    print(f"  CCO vs benzene: {cosine_sim(embeddings[0], embeddings[2]):.4f}")
    print(f"  CCO vs acetic: {cosine_sim(embeddings[0], embeddings[3]):.4f}")

    # Check if embeddings are distinct
    all_same = all(
        cosine_sim(embeddings[0], embeddings[i]) > 0.99
        for i in range(1, len(embeddings))
    )

    if all_same:
        print("\n⚠️  WARNING: All embeddings are nearly identical!")
        print("   Weight loading may not be correct.")
    else:
        print("\n✓ Embeddings are distinct - encoder working correctly!")

    return embeddings


if __name__ == "__main__":
    main()
