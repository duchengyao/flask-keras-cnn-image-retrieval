# Author: S1NH.com

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
ALLOWED_PORT = ('127.0.0.1', 19877)
# INDEX_FILE = "./model/featureCNN_trademark_70k.h5"

DATABASE_IMG_BASE64 = "./database/lctmimage.txt"
INDEX_FILE_TRAINING = "./database/test_base64.h5"

INDEX_FILE = INDEX_FILE_TRAINING

DEBUG = False

MAXRES = 10
QUERY_METHOD = 'faiss_ivfflat'  # {vec_multi, faiss_flat, faiss_ivfflat, faiss_ivfpq}
