# from numpy.lib.function_base import interp
import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import numpy as np
import time
import copy
import os
from model.lanenet.loss import DiscriminativeLoss, FocalLoss
from model.utils import cli_helper
import loguru

LOG = loguru.logger


def compute_loss(net_output,
                 binary_label,
                 instance_label,
                 loss_type='FocalLoss'):
    k_binary = 10  # 1.7
    k_instance = 0.3
    k_dist = 1.0

    if (loss_type == 'FocalLoss'):
        loss_fn = FocalLoss(gamma=2, alpha=[0.25, 0.75])
    elif (loss_type == 'CrossEntropyLoss'):
        loss_fn = nn.CrossEntropyLoss()
    else:
        # print("Wrong loss type, will use the default CrossEntropyLoss")
        loss_fn = nn.CrossEntropyLoss()

    binary_seg_logits = net_output["binary_seg_logits"]
    binary_loss = loss_fn(binary_seg_logits, binary_label)

    pix_embedding = net_output["instance_seg_logits"]
    ds_loss_fn = DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001)
    var_loss, dist_loss, reg_loss = ds_loss_fn(pix_embedding, instance_label)
    binary_loss = binary_loss * k_binary
    var_loss = var_loss * k_instance
    dist_loss = dist_loss * k_dist
    instance_loss = var_loss + dist_loss
    total_loss = binary_loss + instance_loss
    out = net_output["binary_seg_pred"]

    return total_loss, binary_loss, instance_loss, out


def train_model(
        model,
        optimizer,
        scheduler,  # unuse now
        dataloaders,
        dataset_sizes,
        device,
        loss_type='FocalLoss',
        num_epochs=25):
    args = cli_helper.parse_args()
    since = time.time()
    best_loss = float("inf")
    start_epoch = 0
    #  加载断点
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)  # 加载断点
        model.load_state_dict(checkpoint['net'])  # 加载可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch 通过它来保证训练时epoch不会变化
        print('sucess load checkpoint')
        LOG.info('sucess load checkpoint')
    best_model_wts = copy.deepcopy(model.state_dict())
    # 打印学习率
    print('这一阶段学习率为:', optimizer.param_groups[0]['lr'])
    for epoch in range(start_epoch, num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_loss_b = 0.0
            running_loss_i = 0.0

            # Iterate over data.
            for inputs, binarys, instances in dataloaders[phase]:
                inputs = inputs.type(torch.FloatTensor).to(device)
                binarys = binarys.type(torch.LongTensor).to(device)
                instances = instances.type(torch.FloatTensor).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(outputs['binary_seg_logits'].shape) # 输入276这里shape变为了272 为什么？因为Enet的输入得是16的倍数
                    loss = compute_loss(outputs, binarys, instances, loss_type)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss[0].backward()
                        optimizer.step()

                # statistics
                running_loss += loss[0].item() * inputs.size(0)
                running_loss_b += loss[1].item() * inputs.size(0)
                running_loss_i += loss[2].item() * inputs.size(0)

            if phase == 'train':
                if scheduler is None:
                    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            binary_loss = running_loss_b / dataset_sizes[phase]
            instance_loss = running_loss_i / dataset_sizes[phase]
            # print(
            #    '{} Total Loss: {:.4f} Binary Loss: {:.4f} Instance Loss: {:.4f}'
            #    .format(phase, epoch_loss, binary_loss, instance_loss))
            # add log
            log_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                     time.localtime(time.time()))
            LOG.info(
                '=> Epoch: {:d} Time: {:s} {} Total Loss: {:.4f} Binary Loss: {:.4f} Instance Loss: {:.4f}'
                .format(epoch, log_time, phase, epoch_loss, binary_loss,
                        instance_loss))

            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
        print()
        #  保存checkpoint 每隔10个epoch一次
        save_interval = 10
        if (epoch + 1) % save_interval == 0:
            checkpoint = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
            }
            save_time = time.strftime('%Y-%m-%d-%H-%M-%S',
                                      time.localtime(time.time()))
            checkpoint_path = os.path.join(args.save,
                                           save_time + 'checkpoints')
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            torch.save(
                checkpoint, checkpoint_path + '/ckpt_%sepoch_%s_%s.pth' %
                (str(epoch), str(args.model_type), str(args.backend)))
            LOG.info('save checkpoint: ' + checkpoint_path +
                     '/ckpt_%sepoch_%s_%s.pth' %
                     (str(epoch), str(args.model_type), str(args.backend)))
    time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #    time_elapsed // 60, time_elapsed % 60))
    # print('Best val_loss: {:4f}'.format(best_loss))
    LOG.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    LOG.info('Best val_loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    LOG.info('Best model weights saved')
    return model


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable
