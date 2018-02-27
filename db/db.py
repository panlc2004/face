import pymysql.cursors
import numpy as np
import json


def get_conn():
    print('开始建立数据库连接')
    conn = pymysql.connect(host='10.131.0.126',
                           port=3306,
                           user='root',
                           password='mysql3306',
                           db='baggage',
                           charset='utf8',
                           cursorclass=pymysql.cursors.DictCursor)
    return conn


def insert_person_info(name, username, face_data):
    sql = "INSERT INTO person_info (name,username, face_data) VALUES ('" + str(name) + "'" + ",'" + str(
        username) + "','" + str(face_data) + "')"
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        conn.commit()
        insert_id = cursor.lastrowid
    finally:
        cursor.close()
        conn.close()
    return insert_id


def find_all_person_info():
    sql = "SELECT * FROM person_info"
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        result = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()
    return result


def get_person_cache():
    res = find_all_person_info()
    all_data = []
    for r in res:
        face_data = r['face_data']
        load = json.loads(face_data)
        m = np.array(load)
        all_data.append(np.squeeze(m))

    face_data_cache = np.array(all_data)
    return res, face_data_cache
