import torch
import sys
from tqdm import tqdm

def trainer(train_data_loader,optimizer,device,model,weight_module,criterion,last_acc):
    model.train()
    accu_loss_1 = torch.zeros(1).to(device)  # loss_1
    accu_loss_2 = torch.zeros(1).to(device)  # loss_2

    accu_num_1 = torch.zeros(1).to(device)  # accuracy_1
    accu_num_2 = torch.zeros(1).to(device)  # accuracy_2

    optimizer.zero_grad()
    data_loader = tqdm(train_data_loader, file=sys.stdout) #progress bar
    last_acc_1, last_acc_2 = last_acc[0], last_acc[1]
    sample_num = 0

    for step, data in enumerate(data_loader):

        samples, labels = data
        sample_num += samples.shape[0]

        labels_act = labels[:, 0].squeeze()
        labels_loc = labels[:, 1].squeeze()

        pre_act, pre_loc = model(samples.to(device))
        # loss
        loss_act = criterion(pre_act, labels_act)
        loss_loc = criterion(pre_loc, labels_loc)
        loss = weight_module(loss_act, loss_loc, last_acc_1, last_acc_2)
        accu_loss_1 += loss_act.item()
        accu_loss_2 += loss_loc.item()

        # class
        pred_classes_act = torch.max(pre_act, dim=1)[1]
        accu_num_1 += torch.eq(pred_classes_act, labels_act.to(device)).sum()
        pred_classes_loc = torch.max(pre_loc, dim=1)[1]
        accu_num_2 += torch.eq(pred_classes_loc, labels_loc.to(device)).sum()

        loss.backward()

        # data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
        #                                                                        accu_loss.item() / (step + 1),
        #                                                                        accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss_1.item() / (step + 1), accu_num_1.item() / sample_num, accu_loss_2.item() / (step + 1), accu_num_2.item() / sample_num


@torch.no_grad()
def evaluate(val_data_loader,optimizer,device,model,weight_module,criterion,last_acc):
    model.train()
    accu_loss_1 = torch.zeros(1).to(device)  # loss_1
    accu_loss_2 = torch.zeros(1).to(device)  # loss_2

    accu_num_1 = torch.zeros(1).to(device)  # accuracy_1
    accu_num_2 = torch.zeros(1).to(device)  # accuracy_2

    data_loader = tqdm(val_data_loader, file=sys.stdout) #progress bar
    last_acc_1, last_acc_2 = last_acc[0], last_acc[1]
    sample_num = 0

    for step, data in enumerate(data_loader):

        samples, labels = data
        sample_num += samples.shape[0]
        labels_act = labels[:, 0].squeeze()
        labels_loc = labels[:, 1].squeeze()
        pre_act, pre_loc = model(samples.to(device))
        # loss
        loss_act = criterion(pre_act, labels_act)
        loss_loc = criterion(pre_loc, labels_loc)
        loss = weight_module(loss_act, loss_loc, last_acc_1, last_acc_2)
        accu_loss_1 += loss_act.item()
        accu_loss_2 += loss_loc.item()

        # class
        pred_classes_act = torch.max(pre_act, dim=1)[1]
        accu_num_1 += torch.eq(pred_classes_act, labels_act.to(device)).sum()
        pred_classes_loc = torch.max(pre_loc, dim=1)[1]
        accu_num_2 += torch.eq(pred_classes_loc, labels_loc.to(device)).sum()


        # data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
        #                                                                        accu_loss.item() / (step + 1),
        #                                                                        accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss_1.item() / (step + 1), accu_num_1.item() / sample_num, accu_loss_2.item() / (step + 1), accu_num_2.item() / sample_num

