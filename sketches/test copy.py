# -*- coding: UTF-8 -*-
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

    test_dataloader = get_test_data_loader()

    correct = 0
    total = 0
    hits_freq = [0]*MAX_CAPTCHA
    hits_pos = np.array([0]*MAX_CAPTCHA)
    for i, (images, labels) in enumerate(test_dataloader):
        image = images.to(DEVICE)
        vimage = Variable(image)
        predict_label = cnn(vimage)
        # print(predict_label)
        # c0 = ALL_CHAR_SET[np.argmax(predict_label[0, 0:ALL_CHAR_SET_LEN].data.numpy())]
        # c1 = ALL_CHAR_SET[np.argmax(predict_label[0, ALL_CHAR_SET_LEN:2 * ALL_CHAR_SET_LEN].data.numpy())]
        # c2 = ALL_CHAR_SET[np.argmax(predict_label[0, 2 * ALL_CHAR_SET_LEN:3 * ALL_CHAR_SET_LEN].data.numpy())]
        # c3 = ALL_CHAR_SET[np.argmax(predict_label[0, 3 * ALL_CHAR_SET_LEN:4 * ALL_CHAR_SET_LEN].data.numpy())]
        # print(predict_label.shape)
        # print([j for j in range(ALL_CHAR_SET_LEN+1)])
        # predicted_chars = [np.argmax(predict_label[0, (ALL_CHAR_SET_LEN*j):(ALL_CHAR_SET_LEN*(j+1))].data.cpu().numpy()) for j in range(MAX_CAPTCHA)]
        predicted_chars = [ALL_CHAR_SET[np.argmax(predict_label[0, (ALL_CHAR_SET_LEN*j):(ALL_CHAR_SET_LEN*(j+1))].data.cpu().numpy())] for j in range(MAX_CAPTCHA)]
        # c0 = ALL_CHAR_SET[np.argmax(predict_label[0, 0:ALL_CHAR_SET_LEN].data.cpu().numpy())]
        # c1 = ALL_CHAR_SET[np.argmax(predict_label[0, ALL_CHAR_SET_LEN:2 * ALL_CHAR_SET_LEN].data.cpu().numpy())]
        # c2 = ALL_CHAR_SET[np.argmax(predict_label[0, 2 * ALL_CHAR_SET_LEN:3 * ALL_CHAR_SET_LEN].data.cpu().numpy())]
        # c3 = ALL_CHAR_SET[np.argmax(predict_label[0, 3 * ALL_CHAR_SET_LEN:4 * ALL_CHAR_SET_LEN].data.cpu().numpy())]        
        # predict_label_str = '%s%s%s%s' % (c0, c1, c2, c3)
        # predict_label_str = c0
        # predicted_chars = [c0]
        predict_label_str = "".join(predicted_chars)
        true_label = one_hot_encoding.decode(labels.numpy()[0])
        tl_arr = np.array(list(true_label))
        # pl_arr = np.array(["a"]*4) #
        # pl_arr = predict_label.cpu().detach().numpy()[0]
        # print(pl_arr.shape)
        # c0 = ALL_CHAR_SET[np.argmax(pl_arr[0:ALL_CHAR_SET_LEN])]
        # c1 = ALL_CHAR_SET[np.argmax(pl_arr[ALL_CHAR_SET_LEN:2 * ALL_CHAR_SET_LEN])]
        # c2 = ALL_CHAR_SET[np.argmax(pl_arr[2 * ALL_CHAR_SET_LEN:3 * ALL_CHAR_SET_LEN])]
        # c3 = ALL_CHAR_SET[np.argmax(pl_arr[3 * ALL_CHAR_SET_LEN:4 * ALL_CHAR_SET_LEN])]
        # predict_label_str = '%s%s%s%s' % (c0, c1, c2, c3)          
        charswise_comparison = (predicted_chars ==  tl_arr)*1
        # charswise_comparison = (np.array([c0, c1, c2, c3]) == tl_arr)*1
        hit = np.sum(charswise_comparison)
        # print(c0, c1, c2, c3, true_label, hit)
        total += labels.size(0)
        
        hits_freq[max(hit-1, 0)] += 1
        hits_pos+=charswise_comparison
        
        if(predict_label_str == true_label):
            correct += 1
        else:
            print(predict_label_str, true_label)    
            # if hit > best_hit:
            #     best_hit = hit
            #     best_hit_freq = 1
            # elif hit == best_hit:
            #     best_hit_freq += 1
        if(total%200==0):
            # print(labels.shape)
            # print(labels.numpy()[0])
            # print(pl_arr)
            # print(type(predict_label))
            # print(predict_label.shape)
            print(f"{tl_arr}")
            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
            print(f"hits_freq: {hits_freq}; hits_pos: {hits_pos}")
    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))

if __name__ == '__main__':
    main()


