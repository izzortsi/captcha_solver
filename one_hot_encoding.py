# -*- coding: UTF-8 -*-

# %%

import numpy as np
from captcha_settings import *
# %%


char_to_pos = {c: i for i, c in enumerate(ALL_CHAR_SET)}
pos_to_char = {i: c for i, c in enumerate(ALL_CHAR_SET)}

# %%


def encode(text):
    vector = np.zeros(ALL_CHAR_SET_LEN * MAX_CAPTCHA, dtype=float)

    for i, c in enumerate(text):
        idx = i * ALL_CHAR_SET_LEN + char_to_pos[c]
        vector[idx] = 1.0
    return vector

def decode(vec):
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        code = c % ALL_CHAR_SET_LEN
        text.append(pos_to_char[code])
    return "".join(text)

# %%

if __name__ == '__main__':
    cap = "".join(np.random.choice(np.array(ALL_CHAR_SET), MAX_CAPTCHA))
    e = encode(cap)
    print(np.sum(e))
    print(len(e))
    print(cap)
    print(decode(e))
# %%
