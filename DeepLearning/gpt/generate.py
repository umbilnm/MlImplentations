from typing import Callable
from typing import List

import numpy as np

def generate(
    llm: Callable[[List[int]], List[float]],
    prompt: List[str],
    n_tokens: int,
    vocab: List[str],
    top_k: int = 50,
    top_p: float = 0.75,
    temperature: float = 1.1,
    random_state: int = 0,
) -> List[str]:
    """Generate a sequence of tokens from a prompt using a language model."""
    np.random.seed(random_state)

    # TODO: Convert prompt to token IDs. Use 'vocab' to help with this.
    input_ids = list(map(lambda x: vocab.index(x), prompt))
    generated_tokens = []

    # TODO: Auto-regressive generation loop
    for _ in range(n_tokens):
        # TODO: Obtain token probabilities using the callable object
        logits = np.array(llm(input_ids))
        # TODO: Apply temperature, top-p and top-k
        logits = logits/temperature
        probas = np.exp(logits)/(np.exp(logits).sum())
        probas_sorted_k = np.sort(probas)[::-1][:top_k]
        probas_sorted_k = probas_sorted_k/(probas_sorted_k.sum())
        indxs = np.argsort(probas)[::-1][:top_k]
        p_idx = np.where(np.cumsum(probas_sorted_k) >= top_p)[0][0] + 1
        probas_p = probas_sorted_k[:p_idx]
        indxs = indxs[:p_idx]

        # TODO: Sample the next token
        new_idx = np.random.choice(a=indxs, size=1, p = probas_p/(probas_p.sum()))[0]
        generated_tokens.append(vocab[new_idx])
        input_ids.append(new_idx)
    return generated_tokens