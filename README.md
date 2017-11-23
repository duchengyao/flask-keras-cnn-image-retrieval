# Image Retrieval Engine Based on Keras

### Environment

* Anaconda 3
* [Faiss](https://github.com/facebookresearch/faiss)

### Quick start

* git clone https://github.com/lanternfish-research/cnn-based-trademark-retrieval.git
* cd cnn-based-trademark-retrieval
* python create_feature_database_keras.py
* python app.py
* 访问 `http://127.0.0.1:19877/`

```sh
├── confg.py 配置文件
├── app.py flask demo
├── app_api_.py RESTful api
├── database 图像数据集
   ├── lctmimage.txt 图像数据库 ["ID" "base64"]
   └── test_base64.h5 上一个文件的图像特征[[vec],"ID]
├── extract_feat_***.py 使用预训练***模型提取图像特征
|── create_feature_database_keras.py 对图像集提取特征，建立索引
├── query.py 库内搜索
└── README.md
```
