import numpy as np
import os
import scipy.io as sio
from wifilib import *


# 文件设定
dir_name = "D:\csidata\d20181115\d20181115\suser1"
file_list = os.listdir(dir_name)
# 处理函数设定
rate = 15
# 数据信息设定
user = 'user1'
face_orientation = '2'
rx_id = 'r3'
rx_antenna = 1
# 变量初始化
cnt = 0
data_len = 192
train_activity_label = []
train_location_label = []
train_data = []

for name in file_list:
    # name的定义原则：user-ges-loc-face_orienation-repetition_num-rx_id
    # 分割字符串
    name_list = name.split('-')
    dic = {} # 用字典的形式存储字段
    dic['user'] = name_list[0]
    dic['gesture_type'] = name_list[1]
    dic['torso_location'] = name_list[2]
    dic['face_orientation'] = name_list[3]
    dic['repetition_number'] = name_list[4]
    dic['Rx_id'] = name_list[5].split('.')[0]

    if dic['user']==user and dic["face_orientation"]==face_orientation and dic["Rx_id"]==rx_id:
        filename = dir_name + '/' + name

        try:
            data = process_csidata(filename,rate=rate)
        except:
            report = open('report.txt', mode='a')
            report.write(name+'\n')
            report.close()
            print("something wrong",name)
            continue

        train_location_label.append([float(dic['torso_location'])])
        train_activity_label.append([float(dic['gesture_type'])])
        train_data.append(data)
        # print(data.shape)
        cnt += 1
        print("nums:\t",cnt)

train_location_label = np.array(train_location_label)
train_activity_label = np.array(train_activity_label)
train_data = np.array(train_data)
print(train_data.shape)

sio.savemat('data_2.mat',{'train_location_label':train_location_label,'train_activity_label':train_activity_label,'train_data':train_data})