# Author: yongyuan.name
# [1]https://zhuanlan.zhihu.com/p/27101000


import numpy as np
from numpy import linalg as LA
from image_base64 import load_img_base64

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

input_shape = (224, 224, 3)
model = VGG16(weights='imagenet', input_shape=(input_shape[0], input_shape[1], input_shape[2]), pooling='max',
              include_top=False)
model.predict(np.zeros((1, 224, 224, 3)))  # 机器玄学[1]

'''
 Use vgg16 model to extract features
 Output normalized feature vector
'''


def vgg16_feat(img_path=None, img_base64=None):
    # weights: 'imagenet'
    # pooling: 'max' or 'avg'
    # input_shape: (width, height, 3), width and height should >= 48
    if img_path is not None:
        img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    elif img_base64 is not None:
        img = load_img_base64(img_base64, target_size=(input_shape[0], input_shape[1]))
    else:
        return 0

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feat = model.predict(img)
    norm_feat = feat[0] / LA.norm(feat[0])
    return norm_feat
