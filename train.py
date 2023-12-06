import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import hydra
from omegaconf import DictConfig


from utils.LoadModel import *
from utils.train_one_epoch import trainer, evaluate
from utils.dataset import Dataset
# Filtering the warning reports.
import warnings
warnings.filterwarnings('ignore')

@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    # random seeds
    is_seed_fixed = cfg.train_setting.is_seed_fixed
    seeds = cfg.train_setting.seed
    if is_seed_fixed:
        torch.manual_seed(seeds)

    # device settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # callback settings
    tb_writer = SummaryWriter()

    # load data
    train_dataset, num_train_instance, test_dataset, num_test_instance = Dataset(cfg,device)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=cfg.trainer_setting.batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=cfg.trainer_setting.batch_size, shuffle=False)

    # init_the_model
    model = loading_model(cfg)
    weight_module = loading_weight_module(cfg,num_train_instance)

    # init the optimizer.
    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(
        [{'params': model.parameters(), 'lr': cfg.optimizer_setting.model_lr}, {'params': weight_module.parameters(),
        'weight_decay': cfg.optimizer_setting.awl_weight_decay, 'lr': cfg.optimizer_setting.awl_lr}])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10, 20, 30, 40, 60, 70, 80, 90, 100, 110, 120, 130,
                                                                 140, 150, 160, 170, 180, 190, 200],
                                                     gamma=cfg.scheduler_setting.gamma)


    # begin to train the model

    num_epochs = cfg.trainer_setting.num_epochs


    for epoch in range(num_epochs):
        # train
        train_loss_act, train_acc_act,train_loss_loc, train_acc_loc = trainer(train_data_loader=train_data_loader,
                                                                                optimizer=optimizer,
                                                                                device=device,
                                                                                model=model,
                                                                                weight_module=weight_module,
                                                                                criterion=criterion,
                                                                                last_acc=[0, 0]if epoch==0 else [train_acc_act, train_acc_loc] )




        val_loss_act, val_acc_act, val_loss_loc, val_acc_loc = evaluate(val_data_loader=test_data_loader,
                                                                               optimizer=optimizer,
                                                                               device=device,
                                                                               model=model,
                                                                               weight_module=weight_module,
                                                                               criterion=criterion,
                                                                               last_acc=[0, 0] if epoch==0 else [train_acc_act, train_acc_loc])
        scheduler.step()

        print(train_acc_act, train_acc_loc)
        print(val_acc_act,val_acc_loc)
        # validate
        # val_loss, val_acc = evaluate(model=model,
        #                              data_loader=val_loader,
        #                              device=device,
        #                              epoch=epoch)
        #
        tags = ["train_loss", "train_acc_act","train_acc_loc" "val_loss", "val_acc_act", "val_acc_loc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss_act+train_loss_loc, epoch)
        tb_writer.add_scalar(tags[1], train_acc_act, epoch)
        tb_writer.add_scalar(tags[2], train_acc_loc, epoch)
        tb_writer.add_scalar(tags[3], val_loss_act+val_loss_loc, epoch)
        tb_writer.add_scalar(tags[4], val_acc_act, epoch)
        tb_writer.add_scalar(tags[5], val_acc_loc, epoch)
        # tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    main()