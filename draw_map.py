import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
from torch.autograd import Variable
from models.apl import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import defaultdict

def measure(matrix,labels):
    n = len(labels)
    dic = defaultdict(list) # [TP TN FP FN]
    total = sum(sum(matrix))

    for i in range(n):
        TP = matrix[i][i]
        FP = sum(matrix[i]) - TP
        FN = sum(matrix[:][i]) - TP
        TN = total - TP - FP - FN
        # print(TP,FP,FN,TN,total)
        dic['accuracy'].append(100*round((TP+TN)/total,3))
        dic['precision'].append(100*round(TP/(TP+FP),3))
        dic['recall'].append(100*round(TP/(TP+FN),3))
        dic['false_accept_rate'].append(100*round(FP/(FP+TN),3))
        dic['false_reject_rate'].append(100*round(FN/(FN+TP),3))

    return dic





batch_size = 512
data_amp = sio.loadmat('data/test_data_split_amp.mat')
test_data_amp = data_amp['test_data']
test_data = test_data_amp
# data_pha = sio.loadmat('data/test_data_split_pha.mat')
# test_data_pha = data_pha['test_data']
# test_data = np.concatenate((test_data_amp,test_data_pha), 1)

test_activity_label = data_amp['test_activity_label']
test_location_label = data_amp['test_location_label']
test_label = np.concatenate((test_activity_label, test_location_label), 1)

num_test_instances = len(test_data)

test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
test_label = torch.from_numpy(test_label).type(torch.LongTensor)
# test_data = test_data.view(num_test_instances, 1, -1)
# test_label = test_label.view(num_test_instances, 2)

test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

aplnet = ResNet(block=BasicBlock, layers=[1, 1, 1, 1], inchannel=52)
state_dict=torch.load('results_weight/awl_84.pth')
aplnet.load_state_dict(state_dict)
aplnet = aplnet.eval()

correct_test_loc = 0
correct_test_act = 0


act_matrix = np.zeros((6,6),dtype=int)
loc_matrix = np.zeros((16,16),dtype=int)


for i, (samples, labels) in enumerate(test_data_loader):
    with torch.no_grad():
        samplesV = Variable(samples)
        labels_act = labels[:, 0].squeeze()
        labels_loc = labels[:, 1].squeeze()
        labelsV_act = Variable(labels_act)
        labelsV_loc = Variable(labels_loc)

        predict_label_act, predict_label_loc,_,_,_,_,_,_,_ = aplnet(samplesV)
        prediction = predict_label_act.data.max(1)[1]

        correct_test_act += prediction.eq(labelsV_act.data.long()).sum()

        for i in range(len(prediction)):
            act_matrix[labelsV_act.data[i]][prediction[i]] += 1

        print(correct_test_act.cpu().numpy()/num_test_instances)

        prediction = predict_label_loc.data.max(1)[1]
        for i in range(len(prediction)):
            loc_matrix[labelsV_loc.data[i]][prediction[i]] += 1

        correct_test_loc += prediction.eq(labelsV_loc.data.long()).sum()
        print(correct_test_loc.cpu().numpy() / num_test_instances)

print(loc_matrix)
print(act_matrix)


y_true = np.array(['Positive', 'Negative', 'Positive', 'Positive', 'Negative'])
y_pred = np.array(['Positive', 'Positive', 'Negative', 'Positive', 'Negative'])

# 创建混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 定义类别标签
labels_l = ["#"+str(i) for i in range(16)]
labels_a = ['up','down','left','right','circle','cross']

# 创建热力图
plt.figure(num =1)
plt.rcParams['font.size'] = 14
# sns.heatmap(act_matrix/46, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
sns.heatmap(loc_matrix/18, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels_l, yticklabels=labels_l)
# 设置图形属性
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
# 显示混淆矩阵
plt.show()


plt.figure(num =2)
plt.rcParams['font.size'] = 14
sns.heatmap(act_matrix/46, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels_a, yticklabels=labels_a)
# sns.heatmap(loc_matrix/18, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
# 设置图形属性
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
# 显示混淆矩阵
plt.show()

# 计算相关metric

dic_l = measure(matrix=loc_matrix,labels=labels_l)
dic_a = measure(matrix=act_matrix,labels=labels_a)
print(dic_a)
print(dic_l)

plt.figure(num =3)
plt.rcParams['font.size'] = 14
# sns.heatmap(act_matrix/46, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels_a, yticklabels=labels_a)
# sns.heatmap(loc_matrix/18, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
# 设置图形属性
plt.plot(labels_a,dic_a['accuracy'],marker='o',color='r')
plt.title('Accuracy of Activities')
plt.xlabel('Class')
plt.ylabel('Accuracy(%)')
plt.ylim(90,100)
plt.grid()
# 显示混淆矩阵
plt.show()
plt.figure(num=4)
plt.rcParams['font.size'] = 14
# sns.heatmap(act_matrix/46, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels_a, yticklabels=labels_a)
# sns.heatmap(loc_matrix/18, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
# 设置图形属性
plt.plot(labels_l,dic_l['accuracy'],marker='o',color='r')
plt.title('Accuracy of Locations')
plt.xlabel('Class')
plt.ylabel('Accuracy(%)')
plt.ylim(95,100.5)


plt.grid()
# 显示混淆矩阵
plt.show()

# 存储数据
dic = {'act_matrix':act_matrix,'loc_matrix':loc_matrix}
sio.savemat('conmatrix_awl_94_97.mat',dic)
sio.savemat('measure_act.mat',dic_a)
sio.savemat('measure_loc.mat',dic_l)