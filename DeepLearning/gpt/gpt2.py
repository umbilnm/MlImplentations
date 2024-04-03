import numpy as np


def gpt2(inputs, **weights):
    # Extract n_vocab from weights (wte stands for word token embeddings)
    # Given that wte is a matrix of shape (n_vocab, n_embd),
    # we can extract n_vocab as follows:
    vocab_size = weights["wte"].shape[0]

    # Generate logits with random values
    logits = np.random.randn(vocab_size)

    return logits
