# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
from captcha_cnn_model import CNN
from dataset_manager import *
# Hyper Parameters
num_epochs = 20
batch_size = 128 # 64
learning_rate = 0.0001 #0.001
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def main():
    cnn = CNN().to(DEVICE)
    cnn.train()
    print('init net')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the Model
    train_dataloader = get_train_data_loader()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            images = Variable(images.to(DEVICE))
            labels = Variable(labels.float().to(DEVICE))
            predict_labels = cnn(images)
            # print(predict_labels.type)
            # print(labels.type)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.item())
            if (i+1) % 100 == 0:
                torch.save(cnn.state_dict(), f"./model_{MAX_CAPTCHA}c.pkl")   #current is model.pkl
                print("save model")
        print("epoch:", epoch, "step:", i, "loss:", loss.item())
    torch.save(cnn.state_dict(), f"./model_{MAX_CAPTCHA}c.pkl")   #current is model.pkl
    print("save last model")

if __name__ == '__main__':
    main()


