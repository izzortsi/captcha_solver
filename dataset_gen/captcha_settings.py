
# %%

import os

# string.digits + string.ascii_uppercase
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
UPPER_ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
LOWER_ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
SPECIAL_CHARACTERS = list('+@#%=&<>?}{')

# %%


ALL_CHAR_SET = NUMBER + UPPER_ALPHABET + LOWER_ALPHABET + SPECIAL_CHARACTERS
# ALL_CHAR_SET = NUMBER + UPPER_ALPHABET + LOWER_ALPHABET 
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
# %%

MAX_CAPTCHA = 4


IMAGE_HEIGHT = 34
IMAGE_WIDTH = 81

TRAIN_DATASET_PATH = 'dataset' + os.path.sep + 'train'
TEST_DATASET_PATH = 'dataset' + os.path.sep + 'test'
PREDICT_DATASET_PATH = 'dataset' + os.path.sep + 'predict'
RAW_DATASET_PATH = 'dataset' + os.path.sep + 'raw'

# %%
NUM_CROSSES = [2, 3]
NUM_LINES = [3, 5]

FONT_SIZE = 23
FONT = "ArimoBold.ttf"

BG_COLOR = (255, 255, 126)
