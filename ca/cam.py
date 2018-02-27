import cv2
import numpy as np
from scipy import misc
import db.db as db

from detect.face_compare import get_face_bounding_boxes, prewhiten, calculate_face_feature

persons, caches = db.get_person_cache()

cap = cv2.VideoCapture(0)
margin = 32
image_size = 160
while True:
    face_list = []
    ret, frame = cap.read()

    # img = Image.fromarray(frame)
    # img.save('f:/x.jpg')
    # frame = np.array(img)

    bounding_boxes = get_face_bounding_boxes(frame)
    img_size = np.asarray(frame.shape)[0:2]
    nrof_faces = bounding_boxes.shape[0]  # 人脸数目
    print('找到人脸数目为：{}'.format(nrof_faces))

    for box in bounding_boxes:
        box = box.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        # crop = img[face_position[1]:face_position[3],
        #    face_position[0]:face_position[2], ]
        # crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC)

        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(box[0] - margin / 2, 0)
        bb[1] = np.maximum(box[1] - margin / 2, 0)
        bb[2] = np.minimum(box[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(box[3] + margin / 2, img_size[0])
        cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = prewhiten(aligned)
        face_list.append(prewhitened)

    if len(face_list) > 0:
        faces = np.stack(face_list)
        enm = calculate_face_feature(faces)
        score_tmp = np.zeros([enm.shape[0], caches.shape[0], 128])
        for i in range(enm.shape[0]):
            score_tmp[i, :] = enm[i]
        c = np.subtract(caches, score_tmp)
        d = np.square(c)
        s = np.sum(d, axis=2)
        dist = np.sqrt(s)
        sort_index = np.argsort(dist, axis=1)
        boxs = bounding_boxes.astype(int)
        for i in range(sort_index.shape[0]):
            min_index = sort_index[i][0]
            d = dist[i][min_index]
            if d < 0.8:
                name = persons[min_index]['username']
                box = boxs[i]
                cv2.putText(frame, name, (box[0], box[1] - 5), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
