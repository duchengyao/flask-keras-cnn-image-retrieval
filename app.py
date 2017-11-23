# Author: S1NH.com
# https://www.tuicool.com/articles/YNZ7ny
# http://blog.csdn.net/wangjian1204/article/details/76732337

'''
base64的格式（GIF?
'''

import base64
from gevent import monkey
from base64_database import get_base64_from_ID

monkey.patch_all()
import numpy as np
from flask import Flask, request
from gevent import wsgi
import time

from extract_feat_keras import vgg16_feat

import config
import query

ALLOWED_EXTENSIONS = config.ALLOWED_EXTENSIONS
ALLOWED_PORT = config.ALLOWED_PORT

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 最大上传大小 16M

query_vgg16 = query.Query(config.INDEX_FILE_TRAINING)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    v1_page = '<!DOCTYPE html>    <title>Upload New File</title>    <h1>Upload File</h1>    <form action="/" method="POST" enctype="multipart/form-data">    <input type="file" name="file" />    <input type="submit" value="Upload" />    </form>'
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            img_base64 = str(base64.b64encode(file.read()))[2:-1]
            v1_page += '<p> Uploaded Image </br> <img  src="data:image/jpeg;base64,' + img_base64 + '"></p>'
            search_start_time = time.time()
            im_ID_list = [str(im)[2:-1] for im in
                          query_vgg16.action(np.array([vgg16_feat(img_base64=img_base64)]))]
            search_time = time.time() - search_start_time

            v1_page += "<p>------------ [Vgg 16]    Time: " + "%.20f" % search_time + "s. -------------</p>"

            for i, im in enumerate(im_ID_list):
                v1_page += '<img src="data:image/jpeg;base64,' + get_base64_from_ID(im) + '">|'

    v1_page += "<p>Trademark V1 by S1NH</p>"
    return v1_page


if __name__ == "__main__":
    server = wsgi.WSGIServer(ALLOWED_PORT, app)
    server.serve_forever()
