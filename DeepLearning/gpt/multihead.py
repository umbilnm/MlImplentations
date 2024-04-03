import numpy as np


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def softmax(x):
    # We subtract max(x) for numerical stability
    # https://jaykmody.com/blog/stable-softmax/
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    # Calculate the attention scores
    attn_scores = (q @ k.T) / np.sqrt(q.shape[-1])

    # Apply the mask to hide future inputs
    attn_scores += mask

    # Apply softmax to obtain attention weights
    attn_weights = softmax(attn_scores)

    # Weighted sum of values
    output = attn_weights @ v

    return output

def causal_self_attention(x, c_attn, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    x = linear(x, **c_attn)
    q, k, v = np.split(x, 3, axis=-1)

    # Causal mask to hide future inputs
    causal_mask = (1 - np.tri(x.shape[0])) * -1e10

    # Perform causal self-attention
    x = attention(q, k, v, causal_mask)

    x = linear(x, **c_proj)

    return x

def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    x = linear(x, **c_attn)

    # split into qkv
    q, k, v = np.split(x, 3, axis=1)

    # split into heads
    qkv_heads = np.split(q, n_head, axis=1), np.split(k, n_head,axis=1), np.split(v, n_head, axis=1)

    # Causal mask to hide future inputs
    causal_mask = (1 - np.tri(x.shape[0])) * -1e10

    # Perform attention over each head
    out_heads = []
    for i in range(n_head):
        out_head = attention(qkv_heads[0][i], qkv_heads[1][i], qkv_heads[2][i], causal_mask)
        out_heads.append(out_head)

    # Merge heads
    x = np.concatenate(out_heads, axis=-1)

    # Out projection
    x = linear(x, **c_proj)

    return x
