# %%

import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from PIL import Image
import one_hot_encoding as ohe
from captcha_settings import *
import pandas as pd
from torchvision.io import read_image

# %%
# imglabels = pd.read_json(TRAIN_DATASET_LABELS + os.path.sep + "train_labels.json", orient="index")
# imglabels.columns =['label']
# %%

class CustomImageDataset(Dataset):
    def __init__(self, mode="TRAIN", transform=None, target_transform=None):
        if mode == "TRAIN":
            self.mode = mode
            self.img_labels = pd.read_json(TRAIN_DATASET_LABELS + os.path.sep + "train_labels.json", orient="index")
            self.img_labels.columns = ['label']
            self.img_dir = TRAIN_DATASET_PATH
        elif mode == "TEST":
            self.mode = mode
            self.img_labels = pd.read_json(TEST_DATASET_LABELS + os.path.sep + "test_labels.json", orient="index")
            self.img_labels.columns = ['label']
            self.img_dir = TEST_DATASET_PATH
        elif mode == "PREDICT":
            self.mode = mode
            self.img_list = os.listdir(PREDICT_DATASET_PATH)
            # self.img_labels = pd.read_json(TEST_DATASET_LABELS + os.path.sep + "test_labels.json", orient="index")
            self.img_dir = PREDICT_DATASET_PATH            
        # self.img_labels = pd.read_json(TRAIN_DATASET_LABELS + os.path.sep + "train_labels.json", orient="index")
        
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        if self.mode == "PREDICT":
            return len(os.listdir(PREDICT_DATASET_PATH))
        return len(self.img_labels)

    def __getitem__(self, idx):
        if not self.mode == "PREDICT":
            img_path = os.path.join(self.img_dir, str(self.img_labels.index[idx]) + ".png")
        # print(img_path)
        # image = read_image(img_path)
            image = Image.open(img_path)
        # print(image)
            label = self.img_labels.iloc[idx, 0]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label
        else:
            img_path = os.path.join(self.img_dir, self.img_list[idx])
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            return image, img_path
    
transform = transforms.Compose([
    # transforms.ColorJitter(),
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_train_data_loader():
    dataset = CustomImageDataset(mode = "TRAIN", transform=transform, target_transform=ohe.encode)
    return DataLoader(dataset, batch_size=64, shuffle=True)

def get_test_data_loader():
    dataset = CustomImageDataset(mode = "TEST", transform=transform, target_transform=ohe.encode)
    return DataLoader(dataset, batch_size=1, shuffle=True)

def get_predict_data_loader():
    dataset = CustomImageDataset(mode="PREDICT", transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)
#%%

train_dl = get_train_data_loader()
dlds = train_dl.dataset
#%%
dlds.img_labels
#%%
# test_dl = get_test_data_loader()
# dlds = test_dl.dataset
predict_dl = get_predict_data_loader()
#%%
dlds = predict_dl.dataset.img_list
#%%
dlds
# %%
dlds
# %%
# len(dlds)
#%%
# newname = lambda img: img.removeprefix("image").split(".")[0] + ".png"
# imgn = "image0.jpg"
# newname(imgn)
# for i, img in enumerate(os.listdir(PREDICT_DATASET_PATH)):
    # os.rename(PREDICT_DATASET_PATH + os.path.sep + img, PREDICT_DATASET_PATH + os.path.sep + newname(img))
#%%
# for i, img in enumerate(os.listdir(PREDICT_DATASET_PATH)):
    # os.rename(PREDICT_DATASET_PATH + os.path.sep + img, PREDICT_DATASET_PATH + os.path.sep + f"{i}.png")

# %%

#%%
#%%
#%%
