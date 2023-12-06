from models.architecture.model_3_split import *
from models.architecture.model_2_split import *
from models.architecture.model_1_split import *
from models.architecture.cross_stitch import *
from models.architecture.ARIL_model import *
from models.architecture.AMAN_model import *
from models.loss_weight_module.AutomaticWeightedLoss import *

# from utils.conf import WeightModuleConfig


def loading_model(cfg):
    model_name = cfg.model_setting.name
    num_channels = cfg.model_setting.num_channels
    if model_name == "AMAN_model":
        return AMAN_model(block=BasicBlock, layers=[1, 1, 1, 1], inchannel=num_channels)
    elif model_name == "ARIL_model":
        return ARIL(block=BasicBlock, layers=[1, 1, 1, 1], inchannel=num_channels)
    elif model_name == 'model_1_split':
        return Model_1_split(block=BasicBlock, layers=[1, 1, 1, 1], inchannel=num_channels)
    elif model_name == 'model_2_split':
        return Model_2_split(block=BasicBlock, layers=[1, 1, 1, 1], inchannel=num_channels)
    elif model_name == 'mdoel_3_split':
        return Model_3_split(block=BasicBlock, layers=[1, 1, 1, 1], inchannel=num_channels)
    else:
        raise ValueError("cannot match the model name.")


def loading_weight_module(cfg, num):
    model_name = cfg.weight_module_setting.name
    params = cfg.weight_module_setting.params

    if model_name == "AWL_loss":
        return AutomaticWeightedLoss(num=params['num'])
    elif model_name == 'Fixed_Weight_Loss':
        return Fixed_Weight_Loss()
    elif model_name == 'DTP_Loss':
        return DTP_Loss(gamma_0=params['gamma_0'])
    elif model_name == 'DWA_Loss':
        return DWA_Loss(T=params['T'], k=params['k'], num_train_instances=num)
    else:
        raise ValueError("cannot match the model name.")