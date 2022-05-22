# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import dataset_gen.captcha_settings as cs
import my_dataset
from captcha_cnn_model import CNN
import one_hot_encoding

def main():
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('model.pkl'))
    print("load cnn net.")

    test_dataloader = my_dataset.get_test_data_loader()

    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_dataloader):
        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)

        c0 = cs.ALL_CHAR_SET[np.argmax(predict_label[0, 0:cs.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = cs.ALL_CHAR_SET[np.argmax(predict_label[0, cs.ALL_CHAR_SET_LEN:2 * cs.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = cs.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * cs.ALL_CHAR_SET_LEN:3 * cs.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = cs.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * cs.ALL_CHAR_SET_LEN:4 * cs.ALL_CHAR_SET_LEN].data.numpy())]
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        true_label = one_hot_encoding.decode(labels.numpy()[0])
        total += labels.size(0)
        if(predict_label == true_label):
            correct += 1
        if(total%200==0):
            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))

if __name__ == '__main__':
    main()


