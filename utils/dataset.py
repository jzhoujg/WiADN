import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import TensorDataset
from omegaconf import DictConfig

def makedataset(train_dir,data_name,act_label,loc_label,device):

    data_amp = sio.loadmat(train_dir)
    data = data_amp[data_name]
    activity_label = data_amp[act_label]
    location_label = data_amp[loc_label]
    label = np.concatenate((activity_label, location_label), 1)
    num_instances = len(data)
    if device == 'cuda':
        data = torch.from_numpy(data).type(torch.FloatTensor).cuda()
        label = torch.from_numpy(label).type(torch.LongTensor).cuda()
        dataset = TensorDataset(data, label)
    else:
        data = torch.from_numpy(data).type(torch.FloatTensor)
        label = torch.from_numpy(label).type(torch.LongTensor)
        dataset = TensorDataset(data, label)
    return dataset, num_instances

def Dataset(cfg: DictConfig, device: str):


    train_dir = cfg.dataset_setting.train_dataset_dir
    test_dir = cfg.dataset_setting.test_dataset_dir

    train_dataset, num_train_instance = makedataset(train_dir,'train_data','train_activity_label','train_location_label',device)
    test_dataset, num_test_instance = makedataset(test_dir, 'test_data', 'test_activity_label','test_location_label',device)

    return train_dataset, num_train_instance, test_dataset, num_test_instance

