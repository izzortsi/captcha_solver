import numpy as np
import torch
from torch.autograd import Variable
from captcha_settings import *
from dataset_manager import *
from captcha_cnn_model import CNN
import one_hot_encoding
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def main():
    cnn = CNN().to(DEVICE)
    cnn.eval()
    cnn.load_state_dict(torch.load(f'model_{MAX_CAPTCHA}c.pkl'))
    print("load cnn net.")

    predict_dataloader = get_predict_data_loader()

    #vis = Visdom()
    for i, (images, labels) in enumerate(predict_dataloader):
        image = images.to(DEVICE)
        vimage = Variable(image)
        predict_label = cnn(vimage)
        
        predicted_chars = [ALL_CHAR_SET[np.argmax(predict_label[0, (ALL_CHAR_SET_LEN*j):(ALL_CHAR_SET_LEN*(j+1))].data.cpu().numpy())] for j in range(MAX_CAPTCHA)]
        
        predict_label_str = "".join(predicted_chars)
        print(predict_label_str)
        #vis.images(image, opts=dict(caption=c))

if __name__ == '__main__':
    main()


