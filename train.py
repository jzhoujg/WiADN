import scipy.io as sio
import numpy as np
import math
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from models.apl import *
from AutomaticWeightedLoss import *
from collections import defaultdict
import warnings
import argparse

warnings.filterwarnings('ignore')
def train():

    parser = argparse.ArgumentParser('Wi-ADN for dual tasks')
    parser.add_argument('--awl_lr', default=0.05, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()

    # from models.apl_plus import *
    awl_lr = args.awl_lr
    batch_size = args.batch_size
    num_epochs = args.epochs
    # model_save_path = "./weights/awl_"
    # load data

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(12)
    data_amp = sio.loadmat('./data/train_data_split_amp.mat')

    # loading data
    train_data_amp = data_amp['train_data']
    train_data = train_data_amp
    # data_pha = sio.loadmat('data/train_data_split_pha.mat')
    # train_data_pha = data_pha['train_data']
    # train_data = np.concatenate((train_data_amp,train_data_pha),1)
    train_activity_label = data_amp['train_activity_label']
    train_location_label = data_amp['train_location_label']
    train_label = np.concatenate((train_activity_label, train_location_label), 1)
    num_train_instances = len(train_data)
    train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_label = torch.from_numpy(train_label).type(torch.LongTensor)
    # train_data = train_data.view(num_train_instances, 1, -1)
    # train_label = train_label.view(num_train_instances, 2)
    train_dataset = TensorDataset(train_data, train_label)
    # shuffle = True，randomize samples
    data_amp = sio.loadmat('./data/test_data_split_amp.mat')
    test_data_amp = data_amp['test_data']
    test_data = test_data_amp
    test_activity_label = data_amp['test_activity_label']
    test_location_label = data_amp['test_location_label']
    test_label = np.concatenate((test_activity_label, test_location_label), 1)
    num_test_instances = len(test_data)
    test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label = torch.from_numpy(test_label).type(torch.LongTensor)
    test_dataset = TensorDataset(test_data, test_label)


    # data
    aplnet = AMAN(block=BasicBlock, layers=[1, 1, 1, 1], inchannel=52)
    awl = AutomaticWeightedLoss()


    if device == 'cuda':
        aplnet.cuda()
        awl.cuda()
        test_dataset = TensorDataset(test_data.cuda(), test_label.cuda())
        train_dataset = TensorDataset(train_data.cuda(), train_label.cuda())

    print("The device is: {}".format(device))

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(size_average=False).cuda()
    optimizer = torch.optim.Adam([{'params':aplnet.parameters(), 'lr' :0.005},{'params': awl.parameters(), 'weight_decay': 0,'lr':awl_lr}])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     # milestones=[10, 20, 30, 40, 60, 70, 80, 90, 100, 110, 120, 130,
                                                     #             140, 150, 160, 170, 180, 190, 200, 250, 300],
                                                     milestones=[10, 20, 30, 40, 60, 70, 80, 90, 100, 110, 120, 130,
                                                                 140, 150, 160, 170, 180, 190, 200, 250, 300],
                                                     gamma=0.5)

    train_loss_act = np.zeros([num_epochs, 1])
    train_loss_loc = np.zeros([num_epochs, 1])
    test_loss_act = np.zeros([num_epochs, 1])
    test_loss_loc = np.zeros([num_epochs, 1])
    train_acc_act = np.zeros([num_epochs, 1])
    train_acc_loc = np.zeros([num_epochs, 1])
    test_acc_act = np.zeros([num_epochs, 1])
    test_acc_loc = np.zeros([num_epochs, 1])

    acc_test_stat_1 = []
    acc_test_stat_2 = []

    loss_act_train = []
    loss_loc_train = []
    loss_act_val = []
    loss_loc_val = []
    awl_p = []
    acc1, acc2 = 0.05, 0.05


    best_test_act = -1
    com_loc = 0
    best_epoc = 0

    # Training
    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        aplnet.train()
        scheduler.step()
        loss_x = 0
        loss_y = 0


        for (samples, labels) in tqdm(train_data_loader):
            samplesV = Variable(samples)
            labels_act = labels[:, 0].squeeze()
            labels_loc = labels[:, 1].squeeze()
            labelsV_act = Variable(labels_act)
            labelsV_loc = Variable(labels_loc)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            predict_label_act, predict_label_loc,_,_,_,_,_,_,_ = aplnet(samplesV)

            # print(labelsV_act)
            loss_act = criterion(predict_label_act, labelsV_act)
            loss_loc = criterion(predict_label_loc, labelsV_loc)




            loss = awl(loss_act,loss_loc,acc1, acc2)


            loss.backward()
            optimizer.step()

            # loss
            # = loss1+0.5*loss2+0.25*loss3+0.25*loss4
            # loss = loss1+loss2+loss3+loss4

            loss_x += loss_act.item()
            loss_y += loss_loc.item()

            # loss.backward()
            # optimizer.step()




        last_loss_x = loss_x/num_train_instances
        last_loss_y = loss_y/num_train_instances
        loss_act_train.append(loss_x/num_train_instances)
        loss_loc_train.append(loss_y/num_train_instances)


        # print(awl.params)
        awl_p.append(awl.params.cpu().detach().numpy().tolist()[0]/awl.params.cpu().detach().numpy().tolist()[1])
        train_loss_act[epoch] = loss_x / num_train_instances
        train_loss_loc[epoch] = loss_y / num_train_instances

        aplnet.eval()
        # loss_x = 0
        correct_train_act = 0
        correct_train_loc = 0
        for i, (samples, labels) in enumerate(train_data_loader):
            with torch.no_grad():
                samplesV = Variable(samples)
                labels = labels.squeeze()

                labels_act = labels[:, 0].squeeze()
                labels_loc = labels[:, 1].squeeze()
                labelsV_act = Variable(labels_act)
                labelsV_loc = Variable(labels_loc)

                predict_label_act, predict_label_loc,_,_,_,_,_,_,_ = aplnet(samplesV)

                prediction = predict_label_loc.data.max(1)[1]
                correct_train_loc += prediction.eq(labelsV_loc.data.long()).sum()

                prediction = predict_label_act.data.max(1)[1]
                correct_train_act += prediction.eq(labelsV_act.data.long()).sum()

                loss_act = criterion(predict_label_act, labelsV_act)
                loss_loc = criterion(predict_label_loc, labelsV_loc)
                # loss_x += loss.item()

        print("Activity Training accuracy:", (100 * float(correct_train_act) / num_train_instances))
        print("Location Training accuracy:", (100 * float(correct_train_loc) / num_train_instances))
        acc1 = float(correct_train_act) / num_train_instances
        acc2 = float(correct_train_loc) / num_train_instances
        # train_loss[epoch] = loss_x / num_train_instances
        train_acc_act[epoch] = 100 * float(correct_train_act) / num_train_instances
        train_acc_loc[epoch] = 100 * float(correct_train_loc) / num_train_instances



        loss_x = 0
        loss_y = 0
        correct_test_act = 0
        correct_test_loc = 0

        # testing
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

            prediction = predict_label_loc.data.max(1)[1]
            correct_test_loc += prediction.eq(labelsV_loc.data.long()).sum()

            loss_act = criterion(predict_label_act, labelsV_act)
            loss_loc = criterion(predict_label_loc, labelsV_loc)

            loss_x += loss_act.item()
            loss_y += loss_loc.item()

        loss_act_val.append(loss_x/num_test_instances)
        loss_loc_val.append(loss_y/num_test_instances)


        a = 100 * float(correct_test_act) / num_test_instances + 100 * float(correct_test_loc) / num_test_instances
        if a > best_test_act:
            best_test_act = a
            com_act = 100 * float(correct_test_act) / num_test_instances
            com_loc = (100 * float(correct_test_loc) / num_test_instances)
            best_epoc = epoch
            torch.save(aplnet.state_dict(), "./weights/best_model.pth")

        print("Activity Test accuracy:", (100 * float(correct_test_act) / num_test_instances))
        print("Location Test accuracy:", (100 * float(correct_test_loc) / num_test_instances))
        print("The best accuracy: ACT ({}),\t LOC  ({}),\t best epoch ({})".format(com_act,com_loc,best_epoc))
        acc_test_stat_1.append(float(correct_test_act) / num_test_instances)
        acc_test_stat_2.append(float(correct_test_loc) / num_test_instances)

        test_loss_act[epoch] = loss_x / num_test_instances
        test_acc_act[epoch] = 100 * float(correct_test_act) / num_test_instances

        test_loss_loc[epoch] = loss_y / num_test_instances
        test_acc_loc[epoch] = 100 * float(correct_test_loc) / num_test_instances

        testacc_act = str(100 * float(correct_test_act) / num_test_instances)[0:6]
        testacc_loc = str(100 * float(correct_test_loc) / num_test_instances)[0:6]



    # print(loss_act_train)
    # print(loss_act_val)
    # print(loss_loc_train)
    # print(loss_loc_val)
    # print(awl_p)

    # loss_dic = defaultdict(list)
    # loss_dic['loss_act_train'] = loss_act_train
    # loss_dic['loss_act_val'] = loss_act_val
    # loss_dic['loss_loc_train'] = loss_loc_train
    # loss_dic['loss_loc_val'] = loss_loc_val
    # loss_dic['awl_p'] = awl_p
    # sio.savemat('loss.mat',loss_dic)


if __name__ == "__main__":
    train()
