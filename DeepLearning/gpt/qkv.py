import numpy as np


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def attention(q, k, v):
    d_k = q.shape[-1]
    scores = np.matmul(q, k.T) / np.sqrt(d_k)
    weights = softmax(scores)
    output = np.matmul(weights, v)
    return output.tolist()


def self_attention(x, c_attn, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # QKV projections
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # Split into queries, keys, values
    q, k, v = np.split(x, 3, axis=1)

    # Perform self-attention mechanism
    x = attention(q, k, v)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # Output projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd]

    return x