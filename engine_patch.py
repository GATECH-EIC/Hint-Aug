import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
import random
import time

import utils as myutils
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torchattacks
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

def sample_configs(choices, is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False):

    config = {}
    depth = choices['depth']

    if is_visual_prompt_tuning == False and is_adapter == False and is_LoRA == False and is_prefix==False:
        visual_prompt_depth = random.choice(choices['visual_prompt_depth'])
        lora_depth = random.choice(choices['lora_depth'])
        adapter_depth = random.choice(choices['adapter_depth'])
        prefix_depth = random.choice(choices['prefix_depth'])
        config['visual_prompt_dim'] = [random.choice(choices['visual_prompt_dim']) for _ in range(visual_prompt_depth)] + [0] * (depth - visual_prompt_depth)
        config['lora_dim'] = [random.choice(choices['lora_dim']) for _ in range(lora_depth)] + [0] * (depth - lora_depth)
        config['adapter_dim'] = [random.choice(choices['adapter_dim']) for _ in range(adapter_depth)] + [0] * (depth - adapter_depth)
        config['prefix_dim'] = [random.choice(choices['prefix_dim']) for _ in range(prefix_depth)] + [0] * (depth - prefix_depth)

    else:
        if is_visual_prompt_tuning:
            config['visual_prompt_dim'] = [choices['super_prompt_tuning_dim']] * (depth)
        else:
            config['visual_prompt_dim'] = [0] * (depth)
        
        if is_adapter:
             config['adapter_dim'] = [choices['super_adapter_dim']] * (depth)
        else:
            config['adapter_dim'] = [0] * (depth)

        if is_LoRA:
            config['lora_dim'] = [choices['super_LoRA_dim']] * (depth)
        else:
            config['lora_dim'] = [0] * (depth)

        if is_prefix:
            config['prefix_dim'] = [choices['super_prefix_dim']] * (depth)
        else:
            config['prefix_dim'] = [0] * (depth)
        
    return config

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None,is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False, classes=100, gen_model=None, cf_matrix=None, eps=0.01, args=None, pre_soft_mat=None):
    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)
    
    y_out = []
    y_pred = []
    y_true = []
    eps = eps / 255
    alpha = eps * 2 /255
    steps = 1

    atk = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=1)
    atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))

    softmax = nn.Softmax()
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        real_label = targets.clone()
        if args.switch_bn:
            myutils.change_bn(model, 0)
        # freeze_norm(model)
        # sample random config
        if mode == 'baseline':
            config = sample_configs(choices=choices,is_visual_prompt_tuning=is_visual_prompt_tuning,is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=is_prefix)
            # print("current iter config: {}".format(config))
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
            gen_model_module = unwrap_model(gen_model)
            gen_model_module.set_sample_config(config=config)
        elif mode == 'super':
            # sample
            config = sample_configs(choices=choices,is_visual_prompt_tuning=is_visual_prompt_tuning,is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=is_prefix)
            # print("current iter config: {}".format(config))
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
            gen_model_module = unwrap_model(gen_model)
            gen_model_module.set_sample_config(config=config)
        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)

        if mixup_fn is not None:
            copy_samples, copy_targets = samples.clone(), targets.clone()
            copy_targets = torch.empty((targets.shape[0], classes)).fill_(0.1).cuda()
            attack_label = copy_targets.clone()
            for i in range(targets.shape[0]):
                which_classes = targets[i]
                copy_targets[i][which_classes] = 1.0
                if args.use_pre_soft and pre_soft_mat is not None:
                    new_label = softmax(pre_soft_mat[which_classes])
                    attack_label[i] = new_label

            if samples.shape[0] % 2 == 0:
                samples, targets = mixup_fn(samples, targets)
            else:
                targets = copy_targets
        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(samples)
                    loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    if args.patch_fool:
                        outputs, attn = model(samples)
                    else:
                        outputs = model(samples)
                    loss = criterion(outputs, targets)
        else:
            if args.patch_fool:
                outputs, attn = model(samples)
            else:
                outputs = model(samples)
            if teacher_model:
                with torch.no_grad():
                    [teach_output, teacher_attn] = teacher_model(samples)
            else:
                if args.transmix:
                    if isinstance(targets, tuple):  # target is tuple of (target, y1, y2, lam) when switch to cutmix
                        last_attn = torch.mean(attn[-1].detach()[:, :, 0, 1:], dim=1) 
                        targets = mixup_fn.transmix_label(targets, last_attn, samples.shape)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        
        y_out.extend(outputs.detach().cpu().numpy())
        outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(outputs) # Save Prediction
        labels = real_label.detach().cpu().numpy()
        y_true.extend(labels) # Save Truth

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()
        
        torch.cuda.empty_cache()
        # unfreeze_norm(model)
        # Adv images
        if epoch > args.start_adv:
            if args.switch_bn:
                myutils.change_bn(model, 1)
            if cf_matrix is not None:
                cf_matrix = cf_matrix.astype(float)
                for which_class, cf_label in enumerate(cf_matrix):
                    cf_label[which_class] = 0
                    # cf_label.astype(float)
                    label_sum = cf_label.sum()
                    if label_sum > 0:
                        cf_matrix[which_class] = cf_label / float(label_sum)
                new_labels = torch.clone(targets)
                for count, each_label in enumerate(new_labels):
                    which_class = each_label.argmax()
                    if cf_matrix[which_class].sum() != 0:
                        new_labels[count] = torch.from_numpy(cf_matrix[which_class])
                adv_images = atk(samples, new_labels)
                outputs = model(adv_images)
                label_left = copy_targets
            elif args.patch_fool:
                t_attn = torch.nn.functional.normalize(teacher_attn[args.atten_select].mean(dim=1))
                s_attn = torch.nn.functional.normalize(attn[args.atten_select].mean(dim=1))
                t_attn = t_attn.mean(dim=-2)[:, 1:]
                s_attn = s_attn.mean(dim=-2)[:, 1:]
                if s_attn.shape[1] > t_attn.shape[1]:
                    s_attn = s_attn[...,0: t_attn.shape[1]]
                dist_attn = t_attn - s_attn
                my_index = dist_attn.argsort(descending=True)[:, :args.num_patch]
                dist = torch.max(torch.abs(dist_attn))
                if dist > 0.7:
                    adv_images = myutils.patch_fool_fixed(model, copy_samples, attack_label, my_index, args)
                else:
                    if args.use_pre_soft:
                        adv_images = myutils.patch_fool(model, copy_samples, attack_label, args)
                    else:
                        adv_images = myutils.patch_fool(model, copy_samples, copy_targets, args)
                outputs, attn = model(adv_images)
                label_left = copy_targets
            else:
                adv_images = atk(copy_samples, copy_targets)
                outputs = model(adv_images)
                label_left = copy_targets

            loss_syn = criterion(outputs, label_left)
            loss_syn_value = loss_syn.item()
            optimizer.zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            if amp:
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss_scaler(loss_syn, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
            else:
                loss_syn.backward()
                optimizer.step()
            metric_logger.update(loss_syn=loss_syn_value)
            
            torch.cuda.synchronize()
            if model_ema is not None:
                model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    true = np.array(y_true)
    pred = np.array(y_pred)
    # true = np.argmax(true, axis=1)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, y_pred, y_true, y_out

def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis

def match_loss(gw_syn, gw_real, device, dis_metric):
    dis = torch.tensor(0.0).to(device)

    if dis_metric == 'wb':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s'% dis_metric)

    return dis

def freeze_norm(model):
    for name,param in model.named_parameters():
        if 'norm' in name:
            param.requires_grad = False

def unfreeze_norm(model):
    for name,param in model.named_parameters():
        if 'norm' in name:
            param.requires_grad = True

def syn_image(samples, labels, model, device):
    labels = labels.max(1)[1]
    batch_size = samples.shape[0]
    image_left = samples[0: batch_size//2, ...]
    image_right = samples[batch_size//2: , ...]
    label_left = labels[0: batch_size//2, ...]
    label_right = labels[batch_size//2: , ...]
    net_parameters = []
    for param in list(model.parameters()):
        if param.requires_grad:
            net_parameters.append(param)
    
    # import ipdb
    # ipdb.set_trace()
    image_syn = nn.Parameter(image_left.clone(), requires_grad=True)
    optimizer_img = torch.optim.SGD([image_syn, ], lr=0.1, momentum=0.5) # optimizer_img for synthetic data
    optimizer_img.zero_grad()
    criterion = nn.CrossEntropyLoss().to(device)

    # loss = torch.tensor(0.).to(device)
    for i in range(1):
        output_left = model(image_syn)
        loss_left = criterion(output_left, label_left)
        wg_left = torch.autograd.grad(loss_left, net_parameters, create_graph=True)
        
        output_right = model(image_right)
        loss_right = criterion(output_right, label_right)
        wg_right = torch.autograd.grad(loss_right, net_parameters)
        wg_right = list((_.detach().clone() for _ in wg_right))
        # print(wg_right.shape)
        loss = match_loss(wg_left, wg_right, device, 'wb')
    
        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()
    diff = (image_syn.data - image_left.data)

    return image_syn.data

@torch.no_grad()
def evaluate(data_loader, model, device, amp=True, choices=None, mode='super', retrain_config=None,is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False, patch_fool=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        config = sample_configs(choices=choices,is_visual_prompt_tuning=is_visual_prompt_tuning,is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=False)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)


    print("sampled model config: {}".format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print("sampled model parameters: {}".format(parameters))
    
    y_out = []
    y_pred = []
    y_true = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        myutils.change_bn(model, 0)
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            if patch_fool:
                output, attn = model(images)
            else:
                output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
        y_out.extend(output.cpu().numpy())
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        labels = target.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, y_pred, y_true, y_out
