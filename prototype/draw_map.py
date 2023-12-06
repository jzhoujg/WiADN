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
from sklearn.manifold import TSNE

def measure(matrix,labels):
    n = len(labels)
    dic = defaultdict(list) # [TP TN FP FN]
    total = sum(sum(matrix))

    trace = np.trace(matrix)
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

        predict_label_act, predict_label_loc,_,_,_,_,_,feature_act,feature_loc = aplnet(samplesV)
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

print(feature_loc.size())
print(feature_act.size())

feature_loc = torch.flatten(feature_loc,start_dim=1)
feature_act = torch.flatten(feature_act,start_dim=1)

# print(feature_loc.size())
# print(feature_act.size())


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

# T-SNE 画图


plt.figure(num=5)
tsne = TSNE(n_components=2, random_state=33)

# 进行降维
X_tsne = tsne.fit_transform(feature_loc)

# print(len(X_tsne))
# print(labels_act)
# 可视化降维结果
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=labels_loc[:])
plt.title('t-SNE Results of Location Features')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()



plt.figure(num=6)
tsne = TSNE(n_components=2, random_state=21)

# 进行降维
X_tsne = tsne.fit_transform(feature_act)

# print(len(X_tsne))
# print(labels_act)
# 可视化降维结果
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=labels_act[:])
plt.title('t-SNE Results of Activity Features')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.figure(num=7)
loss = sio.loadmat('loss.mat')

epochs = [i+1 for i in range(200)]
ref = [1 for _ in range(200)]
awl_p = loss['awl_p'][0]
awl_p = np.insert(awl_p,0,1)
plt.rcParams['font.size'] = 14
# sns.heatmap(act_matrix/46, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels_a, yticklabels=labels_a)
# sns.heatmap(loc_matrix/18, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
# 设置图形属性
plt.plot(epochs[:150],awl_p[:150],color='r',label='adaptive weight ratio')
plt.plot(epochs[:150],ref[:150],color='b',linestyle='--',label='reference')
plt.title('')
plt.xlabel('epochs')
plt.ylabel('weight 1 / weight 2')
plt.grid()
plt.legend(loc='upper right')
plt.ylim(0.8,1.5)
plt.show()

# print(loss)
#


plt.figure(num=8)
loss = sio.loadmat('loss.mat')

epochs = [i+1 for i in range(199)]
ref = [1 for _ in range(199)]
loss_act_train = loss['loss_act_train'][0]
loss_loc_train = loss['loss_loc_train'][0]
plt.rcParams['font.size'] = 14
# sns.heatmap(act_matrix/46, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels_a, yticklabels=labels_a)
# sns.heatmap(loc_matrix/18, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
# 设置图形属性
plt.plot(epochs[:150],loss_act_train[:150],color='r',label='Location Recognition')
plt.plot(epochs[:150],loss_loc_train[:150],color='b',label='Activity Recognition')
plt.title('')
plt.xlabel('epochs')
plt.ylabel('train loss')
plt.grid()
plt.legend(loc='upper right')
# plt.ylim(0.8,1.5)
plt.show()



# 比较ARIL

ARIL_dic = sio.loadmat('ARIL_matrix.mat')
print(act_matrix-ARIL_dic['act_matrix'])
print(loc_matrix-ARIL_dic['loc_matrix'])



gain_m_act = act_matrix-ARIL_dic['act_matrix']
gain_m_loc = loc_matrix-ARIL_dic['loc_matrix']
act_gain = []
loc_gain = []

for i in range(6):
    act_gain.append(round(gain_m_act[i][i]/46,2))

for i in range(16):
    loc_gain.append(round(gain_m_loc[i][i] / 18, 2))
act_gain[1] = 0.001

plt.figure(num=9)
colors = ['blue', 'orange', 'orange', 'orange', 'orange','orange']
plt.bar(labels_a,act_gain,color=colors)
plt.xlabel('Categories of Activities')
plt.ylabel('Performance Change')
for i, value in enumerate(act_gain):
    if i==0:
        plt.text(i, value, str(value), ha='center', va='top')
    elif i==1:
        plt.text(i, 0, str(0), ha='center', va='bottom')
    else:
        plt.text(i, value, str(value), ha='center', va='bottom')

plt.show()


plt.figure(num=10)
colors = []
for i in range(len(loc_gain)):
    if loc_gain[i]==0:
        loc_gain[i]=0.001
    if loc_gain[i]<0:
        colors.append('blue')
    else:
        colors.append('orange')

plt.bar(labels_l,loc_gain,color=colors)
plt.xlabel('Categories of Locations')
plt.ylabel('Performance Change')
for i, value in enumerate(loc_gain):
    if value==0.001:
        plt.text(i, 0, str(0), ha='center', va='bottom')
    elif value<0:
        plt.text(i, value, str(value), ha='center', va='top')
    else:
        plt.text(i, value, str(value), ha='center', va='bottom')

plt.show()

# 存储数据
dic = {'act_matrix':act_matrix,'loc_matrix':loc_matrix}
sio.savemat('conmatrix_awl_94_97.mat',dic)
sio.savemat('measure_act.mat',dic_a)
sio.savemat('measure_loc.mat',dic_l)
