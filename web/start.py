import base64
import io
from urllib import parse

import cv2
import numpy as np
from keras.preprocessing import image
import time

from PIL import Image
from flask import Flask, request, json, make_response, jsonify
from flask_cors import *
from werkzeug.contrib.fixers import ProxyFix

from detect import face_compare

app = Flask(__name__, static_url_path='')
app.wsgi_app = ProxyFix(app.wsgi_app)
CORS(app)


@app.route("/")
def index():
    return app.send_static_file('index.html')


@app.route("/recognize", methods=['POST'])
def recognize_phone():
    data = request.data
    unquote_data = parse.unquote(data.decode())
    b64_data = str(unquote_data).replace('picture=', '')
    img_decoded = base64.b64decode(b64_data)
    image_data = Image.open(io.BytesIO(img_decoded)).convert("RGB")
    array = image.img_to_array(image_data)

    # array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    # if array.ndim == 2:
    #     array = to_rgb(array)

    t = time.time()
    image_data.save("d:/face_img/" + str(t) + ".jpg")
    res = face_compare.recognize(array)
    print(res)
    return jsonify(res)


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


@app.route("/recognize_web", methods=['POST'])
def recognize_web():
    params = request.form.to_dict()
    encoded_data = params.get('picture')
    unquote_data = parse.unquote(encoded_data)
    b64_data = str(unquote_data).replace('data:image/png;base64,', '')
    img_decoded = base64.b64decode(b64_data)
    image_data = Image.open(io.BytesIO(img_decoded)).convert("RGB")
    array = image.img_to_array(image_data)
    res = face_compare.recognize(array)
    print(res)
    return jsonify(res)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
