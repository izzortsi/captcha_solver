
# %%

import numpy as np
import random
import string
import sys
import os
import json

from PIL import Image, ImageDraw, ImageFont

# %%
from captcha_settings import *
# %%
print(len(os.sys.argv))

FONT = FONTS_PATH + os.path.sep + DEFAULT_FONT
font = ImageFont.truetype(FONT, size=FONT_SIZE)

if len(os.sys.argv) > 2:
    MAX_CAPTCHA = int(os.sys.argv[2])
    # if os.sys.argv[2] == "a":
    #     FONT = FONTS_PATH + os.path.sep +"Arial.ttf"
    # elif os.sys.argv[2] == "ab":
    #     FONT = FONTS_PATH + os.path.sep +"ArimoBold.ttf"
    # elif os.sys.argv[2] == "am":
    #     FONT = FONTS_PATH + os.path.sep +"ArialMedium.ttf"
    # else:
    #     FONT = FONTS_PATH + os.path.sep + DEFAULT_FONT
# fonte usada (bistream vera)

# caminho
# path = RAW_DATASET_PATH
path = TRAIN_DATASET_PATH
def rndPointDisposition(dx, dy):
    """Return random disposition point."""
    x = int(random.uniform(-dx, dx))
    y = int(random.uniform(-dy, dy))
    return (x, y)

def quadPoints(size, disp1, disp2):
    """Return points for QUAD transformation."""
    w, h = size
    x1, y1 = disp1
    x2, y2 = disp2

    return (
        x1,    -y1,
        -x1,    h + y2,
        w + x2, h - y2,
        w - x2, y1)
    
def rndLineTransform(image):
    """Randomly morph Image object with drawn line."""
    w, h = image.size

    # default: 0.3 0.5
    # dx = w * random.uniform(0.2, 0.4)
    # dy = h * random.uniform(0.2, 0.4)
    dx = w * random.uniform(0.04, 0.08)
    dy = h * random.uniform(0.02, 0.06)
    
    x1, y1 = [abs(z) for z in rndPointDisposition(dx, dy)]
    x2, y2 = [abs(z) for z in rndPointDisposition(dx, dy)]

    quad = quadPoints((w, h), (x1, y1), (x2, y2))

    return image.transform(image.size, Image.QUAD,
                            data=quad, resample=Image.BICUBIC, fill=1)

def deform_image(image):
    transformed_image = rndLineTransform(image)

    new_image = Image.new('RGBA', (IMAGE_WIDTH, IMAGE_HEIGHT), color=BG_COLOR)
    new_image.paste(transformed_image, transformed_image)

    return new_image

def draw_cross(ctx, x, y):
    ctx.point((x, y), 'black')
    ctx.point((x+1, y), 'black')
    ctx.point((x-1, y), 'black')
    ctx.point((x, y+1), 'black')
    ctx.point((x, y-1), 'black')

def draw_random_cross(ctx):
    x1 = random.randint(1, IMAGE_WIDTH-1)
    y1 = random.randint(1, IMAGE_HEIGHT-1)

    draw_cross(ctx, x1, y1)

def draw_random_line(ctx):
    x1 = random.randint(0, IMAGE_WIDTH)
    y1 = random.randint(0, IMAGE_HEIGHT)

    x2 = random.randint(0, IMAGE_WIDTH)
    y2 = random.randint(0, IMAGE_HEIGHT)
    ctx.line((x1, y1, x2, y2), 'black')

def draw_random_stuff(ctx):
    num_crosses = random.randint(*NUM_CROSSES)

    for i in range(num_crosses):
        draw_random_cross(ctx)   
    
    num_lines = random.randint(*NUM_LINES)

    for i in range(num_lines):
        draw_random_line(ctx)

def gen_captcha(text):
    # cria uma imagem branca de 190x80
    image = Image.new('RGBA', (IMAGE_WIDTH, IMAGE_HEIGHT), color=BG_COLOR)
    draw = ImageDraw.Draw(image)

    # passo 1: desenha pontos e linhas aleatorias sem deformação e o texto
    draw_random_stuff(draw)
    # draw.text((10, 3), text, spacing = 4, stroke_width = 0, align="center", fill='black', font=font)
    draw.text((IMAGE_WIDTH//2, IMAGE_HEIGHT//2), text, anchor="mm", stroke_width = 0, align="center", fill='black', font=font)
    del draw

    # passo 2: transforma a imagem
    image = deform_image(image)
    bg = Image.new('RGBA', (IMAGE_WIDTH, IMAGE_HEIGHT), color=BG_COLOR)
    
    # image = image.crop(image.getbbox())  # Calculates the bounding box of the non-zero regions in the image
    # image = image.rotate(random.uniform(-10, 10), Image.Resampling.BICUBIC, expand=0)
    
    image = image.rotate(random.randint(-10, 10), resample=Image.Resampling.BICUBIC)
    image = Image.alpha_composite(bg, image)
    # passo 3: repetir passo 1
    draw = ImageDraw.Draw(image)
    draw_random_stuff(draw)
    del draw

    return image

# def gen_string(size=4, chars=string.ascii_lowercase + string.digits):
def gen_string(size=MAX_CAPTCHA, chars=ALL_CHAR_SET):
    return ''.join(random.choice(chars) for _ in range(size))

if len(sys.argv) == 1:
    print("pass the number of samples you want to generate as an argument")
else:

    train_labels = {}
    test_labels = {}
    
    num = int(sys.argv[1])
    train_num = int(num *0.9)
    test_num = num - train_num
    print("train_num:", train_num)
    print("test_num:", test_num)
    assert train_num + test_num == num
    
    print(f"Gerando {num} captchas")
    
    if not os.path.exists(TRAIN_DATASET_PATH):
        os.makedirs(TRAIN_DATASET_PATH)
    if not os.path.exists(TEST_DATASET_PATH):
        os.makedirs(TEST_DATASET_PATH)
    if not os.path.exists(PREDICT_DATASET_PATH):
        os.makedirs(PREDICT_DATASET_PATH)                        
    if not os.path.exists(TRAIN_DATASET_LABELS):
        os.makedirs(TRAIN_DATASET_LABELS)    
    if not os.path.exists(TEST_DATASET_LABELS):
        os.makedirs(TEST_DATASET_LABELS)
            
    for i in range(num):
        if i % 10 == 0:
            print(f"{str(i)} CAPTCHAS gerados")
            
        text = gen_string()
        image = gen_captcha(text)
        filename = f"{i}" + ".png"
        if i <= train_num:
            image.save(TRAIN_DATASET_PATH + os.path.sep + filename)
            train_labels[i] = text
        else:
            image.save(TEST_DATASET_PATH + os.path.sep + filename)            
            test_labels[i] = text            
                
    with open(TRAIN_DATASET_LABELS + os.path.sep + 'train_labels.json', 'w') as fp:
        json.dump(train_labels, fp)        
    with open(TEST_DATASET_LABELS + os.path.sep + 'test_labels.json', 'w') as fp:
        json.dump(test_labels, fp)         