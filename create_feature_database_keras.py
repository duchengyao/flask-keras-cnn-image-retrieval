# Author: yongyuan.name
'''
输出所有图片的向量到h5py数据库, Keras version
'''
import h5py
import numpy as np
import config

from extract_feat_keras import vgg16_feat

DATABASE_IMG_BASE64 = config.DATABASE_IMG_BASE64
INDEX_FILE_TRAINING = config.INDEX_FILE_TRAINING

'''
 Extract features and index the images
'''
if __name__ == "__main__":

    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")

    feats = []
    trademark_id = []

    line_num = 0

    with open(DATABASE_IMG_BASE64, 'r') as f:
        for line in f:
            _id, _, _base64 = line.split("\"")[1:4]
            feats.append(vgg16_feat(img_base64=_base64))
            trademark_id.append(_id)
            print("extracting feature from image No. %d" %(line_num))

            line_num += 1
            if line_num > 7000:
                break

    feats = np.array(feats)
    # directory for storing extracted features
    output = INDEX_FILE_TRAINING

    print(feats)
    print(trademark_id)
    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")

    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data=feats)
    h5f.create_dataset('dataset_2', data=np.string_(trademark_id))

    h5f.close()
