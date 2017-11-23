# Author: S1NH.com

'''
import时载入base64数据库，调用get_base64_from_ID（商标ID）输出商标base64（jpeg）
memory的问题，可能需要重构
'''
import config

DATABASE_IMG_BASE64 = config.DATABASE_IMG_BASE64

trademark_base64 = []
trademark_id = []

print('Reading Image Database...')

with open(DATABASE_IMG_BASE64, 'r') as f:
    for line in f.readlines():
        _id, _, _base64 = line.split("\"")[1:4]
        trademark_base64.append(_base64)
        trademark_id.append(_id)


def get_base64_from_ID(_id):
    db_index = trademark_id.index(_id)
    b64 = trademark_base64[db_index]
    return b64
