
# %%

import numpy as np
import torch
from torch.autograd import Variable
from captcha_settings import *
from dataset_manager import *
from captcha_cnn_model import CNN
import one_hot_encoding
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
# %%
def show_random_images(df, column_name):
    f = plt.figure(figsize=(10,10))
    i=1
    for i in range(16):
        i += 1
        ax = f.add_subplot(4,4,i)
        sample = random.choice(df[column_name])
        image = mpimg.imread(sample)
        ax.set_title(sample.split("/")[-1])
        plt.imshow(image)

# %%
def show_image(path, label):
    f = plt.figure(figsize=(6,6))
    ax = f.add_subplot(1,1,1)
    image = mpimg.imread(path)
    ax.set_title(f"{path.split('/')[-1]} is {label}")
    plt.imshow(image)
# %%
def compare_images(path1=TEST_DATASET_PATH, path2=PREDICT_DATASET_PATH):
    f = plt.figure(figsize=(10,10))
    i=1
    imgs_path1 = os.listdir(path1)
    imgs_path2 = os.listdir(path2)
    for i in range(16):
        i += 1
        if i%2 ==0:
            ax = f.add_subplot(4,4,i)
            img = np.random.choice(imgs_path1)
            sample = os.path.join(path1, img)
            image = mpimg.imread(sample)
            ax.set_title(f"synthetic {img}")
            
            plt.imshow(image)
        else:
            ax = f.add_subplot(4,4,i)
            img = np.random.choice(imgs_path2)
            sample = os.path.join(path2, img)
            image = mpimg.imread(sample)
            ax.set_title(f"organic {img}")
            plt.imshow(image)
    
# %%
compare_images()    
# %%

def main():
    cnn = CNN().to(DEVICE)
    cnn.eval()
    cnn.load_state_dict(torch.load(f'model_{MAX_CAPTCHA}c.pkl'))
    print("load cnn net.")

    predict_dataloader = get_predict_data_loader()
    predictions = []
    # if not os.path.exists(PREDICT_DATASET_LABELS):
    #     os.makedirs(PREDICT_DATASET_LABELS + os.path.sep + "predicted_labels")
    #vis = Visdom()
    for i, (image, img_path) in enumerate(predict_dataloader):
        image = image.to(DEVICE)
        vimage = Variable(image)
        predict_label = cnn(vimage)
        
        predicted_chars = [ALL_CHAR_SET[np.argmax(predict_label[0, (ALL_CHAR_SET_LEN*j):(ALL_CHAR_SET_LEN*(j+1))].data.cpu().numpy())] for j in range(MAX_CAPTCHA)]
        
        predict_label_str = "".join(predicted_chars)
        print(img_path[0], predict_label_str)
        predictions.append(predict_label_str)
        show_image(img_path[0], predict_label_str)
        #vis.images(image, opts=dict(caption=c))

if __name__ == '__main__':
    main()



#%%
