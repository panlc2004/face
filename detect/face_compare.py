import os

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
from tensorflow.python.platform import gfile
import json

import detect.detect_face
from db import db

margin = 32
image_size = 160
persons, caches = db.get_person_cache()
person_res_max_num_config = 3
person_res_max_num = np.minimum(len(persons), person_res_max_num_config)


def load_model(model):
    model_exp = os.path.expanduser(model)
    print('Model filename: %s' % model_exp)
    with gfile.FastGFile(model_exp, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


# 初始化人脸识别神经网络
compare_graph = tf.Graph()
compare_sess = tf.Session()
compare_graph.as_default()
compare_sess.as_default()
print('Creating fact compare networks and loading parameters')
load_model('../model/face_compare.pb')
# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
prediction_placeholder = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

# 初始化人脸检测神经网络
print('Creating fact detect networks and loading parameters')
pnet, rnet, onet = detect.detect_face.create_mtcnn(compare_sess, None)


def calculate_face_feature(faces):
    feed_dict = {images_placeholder: faces, phase_train_placeholder: False}
    emb = compare_sess.run(prediction_placeholder, feed_dict=feed_dict)
    return emb


def calculate_img_feature(images_files, image_size=160, margin=32):
    images = get_face(images_files, image_size, margin)
    return calculate_face_feature(images)


def get_face(image_paths, image_size, margin):
    # minsize = 20  # minimum size of face
    # threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    # factor = 0.709  # scale factor
    tmp_image_paths = image_paths.copy()
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 2:
            img = to_rgb(img)
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes = get_face_bounding_boxes(img)
        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            print("can't detect face, remove ", image)
            continue
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def get_face_bounding_boxes(img, minsize=20, threshold=[0.6, 0.7, 0.7], factor=0.709):
    bounding_boxes, _ = detect.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    return bounding_boxes


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def recognize(picture):
    face_list = []
    rt_faces = []
    bounding_boxes = get_face_bounding_boxes(picture)
    img_size = np.asarray(picture.shape)[0:2]
    print(img_size)
    for box in bounding_boxes:
        box = box.astype(int)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(box[0] - margin / 2, 0)
        bb[1] = np.maximum(box[1] - margin / 2, 0)
        bb[2] = np.minimum(box[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(box[3] + margin / 2, img_size[0])
        # 添加返回信息
        rt_faces.append({"x": str(bb[0]), "y": str(bb[1]),
                         "w": str(bb[2] - bb[0]), "h": str(bb[3] - bb[1])})
        # 收集脸部区域图像信息，以供识别
        cropped = picture[bb[1]:bb[3], bb[0]:bb[2], :]

        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = prewhiten(aligned)
        face_list.append(prewhitened)

    if len(face_list) > 0:
        faces = np.stack(face_list)
        enm = calculate_face_feature(faces)
        score_tmp = np.zeros([enm.shape[0], caches.shape[0], 128])
        for i in range(enm.shape[0]):
            score_tmp[i, :] = enm[i]
        diff = np.subtract(caches, score_tmp)
        square = np.square(diff)
        dist = np.sum(square, axis=2)
        print('dist: ', dist)
        # dist = np.sqrt(s)

        sort_index = np.argsort(dist, axis=1)

        for i in range(sort_index.shape[0]):
            person_info = []
            for j in range(person_res_max_num):
                _index = sort_index[i][j]
                _d_i = dist[i][_index]
                username = persons[_index]['username']
                person_info.append({"PersonFlag": username, "Confidence": calculate_percent(_d_i)})
            rt_faces[i]["Results"] = person_info

    res = {"opResult": "true", "opMsg": None, "rtFaces": rt_faces}
    return res


def calculate_percent(dist):
    threshold = [1.15, 1.21, 1.23, 1.25]
    return 1 - dist / 2
    # if dist <= 0.5:
    #     return 1 - np.power(dist, 4)
    # return np.sum(np.multiply(dist, -1.0764), 1.5069)


# res1 = calculate_img_feature(['E:/学习资料/人脸识别/face_data/plc.jpg'])
# db.insert_person_info('潘乐春', '003914', json.dumps(res1.tolist()))

#
# res1 = calculate_img_feature(['E:/学习资料/人脸识别/face_data/身份证头像/lgf2.jpg'])
# db.insert_person_info('李葛夫', '001921_2', json.dumps(res1.tolist()))

#
# res1 = calculate_img_feature(['D:/face_img/zy1.jpg'])
# db.insert_person_info('张洋', '011424', json.dumps(res1.tolist()))
# res1 = calculate_img_feature(['D:/face_img/lgf1.jpg'])
# db.insert_person_info('李葛夫', '001921', json.dumps(res1.tolist()))
# res1 = calculate_img_feature(['D:/face_img/lgt1.jpg'])
# db.insert_person_info('刘广涛', '004961', json.dumps(res1.tolist()))
# res1 = calculate_img_feature(['D:/face_img/plc1.jpg'])
# db.insert_person_info('潘乐春', '003914', json.dumps(res1.tolist()))
# res1 = calculate_img_feature(['D:/face_img/mxs4.jpg'])
# db.insert_person_info('莫晓松', '003819', json.dumps(res1.tolist()))
