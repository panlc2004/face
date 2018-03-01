import os

import numpy as np
import tensorflow as tf
from scipy import misc
from tensorflow.python.platform import gfile

import detect.detect_face


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
# face_detect_graph = tf.Graph()
# face_detect_sess = tf.Session()
# face_detect_graph.as_default()
# face_detect_sess.as_default()
print('Creating fact detect networks and loading parameters')
pnet, rnet, onet = detect.detect_face.create_mtcnn(compare_sess, None)


def calculate_face_feature(faces):
    feed_dict = {images_placeholder: faces, phase_train_placeholder: False}
    emb = compare_sess.run(prediction_placeholder, feed_dict=feed_dict)
    return emb


def calculate_img_feature(images_files, image_size=160, margin=32):
    images = get_face(images_files, image_size, margin)
    # predict
    # compare_graph.as_default()
    # compare_sess.as_default()
    return calculate_face_feature(images)


def get_face(image_paths, image_size, margin):
    # minsize = 20  # minimum size of face
    # threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    # factor = 0.709  # scale factor

    # face_detect_graph.as_default()
    # face_detect_sess.as_default()

    tmp_image_paths = image_paths.copy()
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
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


def get_face_bounding_boxes(img, minsize=20, threshold=[0.6, 0.7, 0.7], factor=0.709):
    bounding_boxes, _ = detect.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    return bounding_boxes


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

# res1 = calculate_img_feature(['E:/学习资料/人脸识别/face_data/谢林志/data.jpg'])
# db.insert_person_info('谢林志', '007', json.dumps(res1.tolist()))

# res1 = calculate_img_feature(['E:/学习资料/人脸识别/face_data/plc.jpg'])
# db.insert_person_info('潘乐春', '003914', json.dumps(res1.tolist()))

#
# res1 = calculate_img_feature(['E:/学习资料/人脸识别/face_data/身份证头像/lgf2.jpg'])
# db.insert_person_info('李葛夫', '001921_2', json.dumps(res1.tolist()))


# res1 = calculate_img_feature(['E:/学习资料/人脸识别/face_data/身份证头像/zy2.jpg'])
# db.insert_person_info('张洋', '011424_2', json.dumps(res1.tolist()))