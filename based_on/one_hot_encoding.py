# -*- coding: UTF-8 -*-

# %%

import numpy as np
import captcha_settings as cs

# %%

char_to_pos = {c: i for i, c in enumerate(cs.ALL_CHAR_SET)}
pos_to_char = {i: c for i, c in enumerate(cs.ALL_CHAR_SET)}

# %%


def encode(text):
    vector = np.zeros(cs.ALL_CHAR_SET_LEN * cs.MAX_CAPTCHA, dtype=float)

    for i, c in enumerate(text):
        idx = i * cs.ALL_CHAR_SET_LEN + char_to_pos[c]
        vector[idx] = 1.0
    return vector

def decode(vec):
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        code = c % cs.ALL_CHAR_SET_LEN
        text.append(pos_to_char[code])
    return "".join(text)

# %%

if __name__ == '__main__':
    e = encode("@k7%")
    print(len(e))
    print(e)
    print(decode(e))
# %%
