# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
#from visdom import Visdom # pip install Visdom
import dataset_gen.captcha_settings as cs
import my_dataset
from captcha_cnn_model import CNN

def main():
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('model.pkl'))
    print("load cnn net.")

    predict_dataloader = my_dataset.get_predict_data_loader()

    #vis = Visdom()
    for i, (images, labels) in enumerate(predict_dataloader):
        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)

        c0 = cs.ALL_CHAR_SET[np.argmax(predict_label[0, 0:cs.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = cs.ALL_CHAR_SET[np.argmax(predict_label[0, cs.ALL_CHAR_SET_LEN:2 * cs.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = cs.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * cs.ALL_CHAR_SET_LEN:3 * cs.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = cs.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * cs.ALL_CHAR_SET_LEN:4 * cs.ALL_CHAR_SET_LEN].data.numpy())]

        c = '%s%s%s%s' % (c0, c1, c2, c3)
        print(c)
        #vis.images(image, opts=dict(caption=c))

if __name__ == '__main__':
    main()


