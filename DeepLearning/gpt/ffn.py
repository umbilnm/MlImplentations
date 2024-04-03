import numpy as np

def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x@w + b


def gelu(x):
    cdf = 0.5 * (1.0 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))))
    return x * cdf

def ffn(x, c_fc, c_proj):
    out = linear(x=x, **c_fc)
    out = gelu(out)
    out = linear(out, **c_proj)
    return out