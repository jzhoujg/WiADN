import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from make_model import *

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from make_model import Three_D_Model as create_model
from utils import read_split_data, train_one_epoch, evaluate
import torch.nn as nn


from my_dataset import MyDataSet


# def test(model,args):
#     kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#     if args.dataset == 'cifar10':
#         test_loader = torch.utils.data.DataLoader(
#             datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
#             batch_size=args.test_batch_size, shuffle=True, **kwargs)
#     elif args.dataset == 'cifar100':
#         test_loader = torch.utils.data.DataLoader(
#             datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
#             batch_size=args.test_batch_size, shuffle=True, **kwargs)
#     else:
#         raise ValueError("No valid dataset is given.")
#     model.eval()
#     correct = 0
#     for data, target in test_loader:
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data), Variable(target)
#         output = model(data)
#         pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#
#     print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
#         correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
#     return correct / float(len(test_loader.dataset))

# Prune settings


def makeindexes(num,idx):
    inx_n = []
    for i in idx:
        for j in range((16)):

            inx_n.append(j+16*(i))
    print(inx_n)
    return inx_n


def main(args):

    # 配置显卡
    cfg =  [26, 'M', 30, 'M', 39, 'M']

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path) #保持创建文件夹的方式就行


    data_transform = {
        "train": transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])]),

        "val": transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5], [0.5])])}


    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))



    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = "cuda:0" if args.cuda else "cpu"
    # 修剪后的路径
    if not os.path.exists(args.save):
        os.makedirs(args.save)


    # 模型
    model = Three_D_Model(=cfg)
    last_layer_num = 39
    if args.cuda:
        model.cuda()

    if args.weights:
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        print(model.load_state_dict(weights_dict, strict=False))
    else:
        print("no model to prune")

    print(model)

    total = 0


    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size


    # 进行阈值的计算
    y, i = torch.sort(bn)
    thre_index = int(total * args.percent)
    thre = y[thre_index]

    # 完成
    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm3d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float() # gt 指的是大于或者等于的相关的操作
            pruned = pruned + mask.shape[0] - torch.sum(mask) # 计算小于或者等于
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool3d):
            cfg.append('M')

    pruned_ratio = pruned/total

    print('Pre-processing Successful!')

    # simple test model after Pre-processing prune (simple set BN scales to zeros)

    val_loss, val_acc = evaluate(model=model,
                                 data_loader=val_loader,
                                 device=device,
                                 epoch=200)
    # acc = test(model)

    # Make real prune
    print(cfg)
    newmodel = Three_D_Model(cfg=cfg)
    if args.cuda:
        newmodel.cuda()

    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    savepath = os.path.join(args.save, "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n"+str(cfg)+"\n")
        fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
        fp.write("Test accuracy: \n"+str(val_acc ))

    layer_id_in_cfg = 0
    start_mask = torch.ones(1)
    end_mask = cfg_mask[layer_id_in_cfg]

    # 将该Model
    for [m0, m1] in zip(model.modules(), newmodel.modules()):

        if isinstance(m0, nn.BatchNorm3d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv3d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :,:].clone()
            m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            idx_l = makeindexes(last_layer_num,idx0)
            m1.weight.data = m0.weight.data[:, idx_l].clone()
            m1.bias.data = m0.bias.data.clone()
            break

    torch.save(newmodel.state_dict(), os.path.join(args.save, 'pruned.pth'))


    print(newmodel)
    model = newmodel
    # test(model,args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Slimming BVP prune')

    parser.add_argument('--weights', type=str, default='last.pth', help='initial weights path')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--percent', type=float, default=0.5,
                        help='scale sparse rate (default: 0.)')

    parser.add_argument('--save', default='./prune/', type=str, metavar='PATH',
                        help='path to save pruned model (default: none)')
    parser.add_argument('--data-path', type=str,
                        default="./BVP_MAP")
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    main(args)


