# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from model.supernet_vision_transformer_timm_switch import VisionTransformer, switchableNorm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

import numpy as np
import matplotlib.pyplot as plt
import math

def normalize_vec(vec):
    return (vec - vec.min()) / (vec.max() - vec.min())

def change_bn(model, flag):
    for m in model.modules():
        if isinstance(m, switchableNorm):
            m.set_norm(flag)


def replace_bn(model, orig_type, new_type, copy_param=True):
    for name, module in model.named_children():
        if len(list(module.children()))>0:
            replace_bn(module, orig_type, new_type)
        if isinstance(module, orig_type) and not isinstance(module, new_type):
            # replace norm with copy 
            param_list = {}
            orig_layer = getattr(model, name)
            weight = orig_layer.weight
            bias = orig_layer.bias 
            channel = module.weight.shape[0]
            if hasattr(orig_layer, 'running_mean'):
                rm = orig_layer.running_mean 
                rv = orig_layer.running_var 
            
            new_bn = new_type(channel)
            assert isinstance(new_bn, switchableNorm)

            new_bn.norm[0].weight = weight 
            new_bn.norm[0].bias = bias 

            new_bn.norm[1].weight = weight 
            new_bn.norm[1].bias = bias 

            setattr(model, name, new_bn)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

'''
@Parameter atten_grad, ce_grad: should be 2D tensor with shape [batch_size, -1]
'''
def PCGrad(atten_grad, ce_grad, sim, shape):
    pcgrad = atten_grad[sim < 0]
    temp_ce_grad = ce_grad[sim < 0]
    dot_prod = torch.mul(pcgrad, temp_ce_grad).sum(dim=-1)
    dot_prod = dot_prod / torch.norm(temp_ce_grad, dim=-1)
    pcgrad = pcgrad - dot_prod.view(-1, 1) * temp_ce_grad
    atten_grad[sim < 0] = pcgrad
    atten_grad = atten_grad.view(shape)
    return atten_grad

def get_attn_mask(model, X, y, args):
    patch_size = 16    
    filter = torch.ones([1, 3, patch_size, patch_size]).float().cuda()

    patch_num_per_line = int(X.size(-1) / patch_size)
    delta = torch.zeros_like(X).cuda()
    delta.requires_grad = True

    model.zero_grad()
    out, atten = model(X)
    
    '''choose patch'''
    # max_patch_index size: [Batch, num_patch attack]
    if args.patch_select == 'Rand':
        '''random choose patch'''
        max_patch_index = np.random.randint(0, 14 * 14, (X.size(0), args.num_patch))
        max_patch_index = torch.from_numpy(max_patch_index)
    elif args.patch_select == 'Saliency':
        '''gradient based method'''
        grad = torch.autograd.grad(loss, delta)[0]
        # print(grad.shape)
        grad = torch.abs(grad)
        patch_grad = F.conv2d(grad, filter, stride=patch_size)
        patch_grad = patch_grad.view(patch_grad.size(0), -1)
        max_patch_index = patch_grad.argsort(descending=True)[:, :args.num_patch]
    elif args.patch_select == 'Attn':
        '''attention based method'''
        atten_layer = atten[args.atten_select].mean(dim=1)
        atten_layer = atten_layer.mean(dim=-2)[:, 1:]
        max_patch_index = atten_layer.argsort(descending=True)[:, :args.num_patch]
    else:
        print(f'Unknown patch_select: {args.patch_select}')
        raise
    '''build mask'''
    mask = torch.zeros([X.size(0), 1, X.size(2), X.size(3)]).cuda()
    if args.sparse_pixel_num != 0:
        learnable_mask = mask.clone()

    for j in range(X.size(0)):
        index_list = max_patch_index[j]
        for index in index_list:
            row = (index // patch_num_per_line) * patch_size
            column = (index % patch_num_per_line) * patch_size

            if args.sparse_pixel_num != 0:
                learnable_mask.data[j, :, row:row + patch_size, column:column + patch_size] = torch.rand(
                    [patch_size, patch_size])
            mask[j, :, row:row + patch_size, column:column + patch_size] = 1
    return mask

def patch_fool_mask(model, X, y, args):
    patch_size = 16    
    filter = torch.ones([1, 3, patch_size, patch_size]).float().cuda()
    mu = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1).cuda()
    std = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1).cuda()

    patch_num_per_line = int(X.size(-1) / patch_size)
    delta = torch.zeros_like(X).cuda()
    delta.requires_grad = True

    model.zero_grad()
    out, atten = model(X + delta)
    
    '''choose patch'''
    # max_patch_index size: [Batch, num_patch attack]
    if args.patch_select == 'Rand':
        '''random choose patch'''
        max_patch_index = np.random.randint(0, 14 * 14, (X.size(0), args.num_patch))
        max_patch_index = torch.from_numpy(max_patch_index)
    elif args.patch_select == 'Saliency':
        '''gradient based method'''
        grad = torch.autograd.grad(loss, delta)[0]
        # print(grad.shape)
        grad = torch.abs(grad)
        patch_grad = F.conv2d(grad, filter, stride=patch_size)
        patch_grad = patch_grad.view(patch_grad.size(0), -1)
        max_patch_index = patch_grad.argsort(descending=True)[:, :args.num_patch]
    elif args.patch_select == 'Attn':
        '''attention based method'''
        atten_layer = atten[args.atten_select].mean(dim=1)
        atten_layer = atten_layer.mean(dim=-2)[:, 1:]
        max_patch_index = atten_layer.argsort(descending=True)[:, :args.num_patch]
    else:
        print(f'Unknown patch_select: {args.patch_select}')
        raise
    '''build mask'''
    mask = torch.zeros([X.size(0), 1, X.size(2), X.size(3)]).cuda()
    if args.sparse_pixel_num != 0:
        learnable_mask = mask.clone()

    for j in range(X.size(0)):
        index_list = max_patch_index[j]
        for index in index_list:
            row = (index // patch_num_per_line) * patch_size
            column = (index % patch_num_per_line) * patch_size

            if args.sparse_pixel_num != 0:
                learnable_mask.data[j, :, row:row + patch_size, column:column + patch_size] = torch.rand(
                    [patch_size, patch_size])
            mask[j, :, row:row + patch_size, column:column + patch_size] = 1

    '''adv attack'''
    max_patch_index_matrix = max_patch_index[:, 0]
    max_patch_index_matrix = max_patch_index_matrix.repeat(197, 1)
    max_patch_index_matrix = max_patch_index_matrix.permute(1, 0)
    max_patch_index_matrix = max_patch_index_matrix.flatten().long()

    if args.mild_l_inf == 0:
        '''random init delta'''
        delta = (torch.rand_like(X) - mu) / std
    else:
        '''constrain delta: range [x-epsilon, x+epsilon]'''
        epsilon = args.mild_l_inf / std
        delta = 2 * epsilon * torch.rand_like(X) - epsilon + X

    delta.data = clamp(delta, (0 - mu) / std, (1 - mu) / std)
    original_img = X.clone()
    if args.random_sparse_pixel:
        '''random select pixels'''
        sparse_mask = torch.zeros_like(mask)
        learnable_mask_temp = learnable_mask.view(learnable_mask.size(0), -1)
        sparse_mask_temp = sparse_mask.view(sparse_mask.size(0), -1)
        value, _ = learnable_mask_temp.sort(descending=True)
        threshold = value[:, args.sparse_pixel_num - 1].view(-1, 1)
        sparse_mask_temp[learnable_mask_temp >= threshold] = 1
        mask = sparse_mask

    if args.sparse_pixel_num == 0 or args.random_sparse_pixel:
        X = torch.mul(X, 1 - mask)
    else:
        '''select by learnable mask'''
        learnable_mask.requires_grad = True
    delta = delta.cuda()
    delta.requires_grad = True

    opt = torch.optim.Adam([delta], lr=args.attack_learning_rate)
    if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
        mask_opt = torch.optim.Adam([learnable_mask], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)

    '''Start Adv Attack'''
    for train_iter_num in range(args.train_attack_iters):
        model.zero_grad()
        opt.zero_grad()

        '''Build Sparse Patch attack binary mask'''
        if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
            if train_iter_num < args.learnable_mask_stop:
                mask_opt.zero_grad()
                sparse_mask = torch.zeros_like(mask)
                learnable_mask_temp = learnable_mask.view(learnable_mask.size(0), -1)
                sparse_mask_temp = sparse_mask.view(sparse_mask.size(0), -1)
                value, _ = learnable_mask_temp.sort(descending=True)

                threshold = value[:, args.sparse_pixel_num-1].view(-1, 1)
                sparse_mask_temp[learnable_mask_temp >= threshold] = 1

                '''inference as sparse_mask but backward as learnable_mask'''
                temp_mask = ((sparse_mask - learnable_mask).detach() + learnable_mask) * mask
            else:
                temp_mask = sparse_mask

            X = original_img * (1-sparse_mask)        
            out, atten = model(X + torch.mul(delta, temp_mask))
 
        else:
            out, atten = model(X + torch.mul(delta, mask))

        criterion = nn.CrossEntropyLoss().cuda()
        '''final CE-loss'''

        loss = criterion(out, y)

        if args.attack_mode == 'Attention':
            grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
            ce_loss_grad_temp = grad.view(X.size(0), -1).detach().clone()
            if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                mask_grad = torch.autograd.grad(loss, learnable_mask, retain_graph=True)[0]

            # Attack the first 6 layers' Attn
            range_list = range(len(atten)//2)
            for atten_num in range_list:
                if atten_num == 0:
                    continue
                atten_map = atten[atten_num]
                atten_map = atten_map.mean(dim=1)
                atten_map = atten_map.view(-1, atten_map.size(-1))
                atten_map = -torch.log(atten_map)

                atten_loss = F.nll_loss(atten_map, max_patch_index_matrix + 1)
                atten_grad = torch.autograd.grad(atten_loss, delta, retain_graph=True)[0]

                atten_grad_temp = atten_grad.view(X.size(0), -1)
                cos_sim = F.cosine_similarity(atten_grad_temp, ce_loss_grad_temp, dim=1)

                if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                    mask_atten_grad = torch.autograd.grad(atten_loss, learnable_mask, retain_graph=True)[0]

                '''PCGrad'''
                atten_grad = PCGrad(atten_grad_temp, ce_loss_grad_temp, cos_sim, grad.shape)
                if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
                    mask_atten_grad_temp = mask_atten_grad.view(mask_atten_grad.size(0), -1)
                    ce_mask_grad_temp = mask_grad.view(mask_grad.size(0), -1)
                    mask_cos_sim = F.cosine_similarity(mask_atten_grad_temp, ce_mask_grad_temp, dim=1)
                    mask_atten_grad = PCGrad(mask_atten_grad_temp, ce_mask_grad_temp, mask_cos_sim, mask_atten_grad.shape)

                grad += atten_grad * args.atten_loss_weight
                if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
                    mask_grad += mask_atten_grad * args.atten_loss_weight
        else:
            '''no attention loss'''
            if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
                mask_grad = torch.autograd.grad(loss, learnable_mask)[0]
            else:
                grad = torch.autograd.grad(loss, delta)[0]
        opt.zero_grad()
        delta.grad = -grad
        opt.step()
        scheduler.step()

        if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
            mask_opt.zero_grad()
            learnable_mask.grad = -mask_grad
            mask_opt.step()

            learnable_mask_temp = learnable_mask.view(X.size(0), -1)
            learnable_mask.data -= learnable_mask_temp.min(-1)[0].view(-1, 1, 1, 1)
            learnable_mask.data += 1e-6
            learnable_mask.data *= mask

        '''l2 constrain'''
        if args.mild_l_2 != 0:
            radius = (args.mild_l_2 / std).squeeze()
            perturbation = (delta.detach() - original_img) * mask
            l2 = torch.linalg.norm(perturbation.view(perturbation.size(0), perturbation.size(1), -1), dim=-1)
            radius = radius.repeat([l2.size(0), 1])
            l2_constraint = radius / l2
            l2_constraint[l2 < radius] = 1.
            l2_constraint = l2_constraint.view(l2_constraint.size(0), l2_constraint.size(1), 1, 1)
            delta.data = original_img + perturbation * l2_constraint

        '''l_inf constrain'''
        if args.mild_l_inf != 0:
            epsilon = args.mild_l_inf / std
            delta.data = clamp(delta, original_img - epsilon, original_img + epsilon)

        delta.data = clamp(delta, (0 - mu) / std, (1 - mu) / std)
    
    perturb_x = X + torch.mul(delta, mask)

    return perturb_x, mask

def patch_fool(model, X, y, args):
    patch_size = 16    
    filter = torch.ones([1, 3, patch_size, patch_size]).float().cuda()
    mu = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1).cuda()
    std = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1).cuda()

    patch_num_per_line = int(X.size(-1) / patch_size)
    delta = torch.zeros_like(X).cuda()
    delta.requires_grad = True

    model.zero_grad()
    out, atten = model(X + delta)
    
    '''choose patch'''
    # max_patch_index size: [Batch, num_patch attack]
    if args.patch_select == 'Rand':
        '''random choose patch'''
        max_patch_index = np.random.randint(0, 14 * 14, (X.size(0), args.num_patch))
        max_patch_index = torch.from_numpy(max_patch_index)
    elif args.patch_select == 'Saliency':
        '''gradient based method'''
        grad = torch.autograd.grad(loss, delta)[0]
        # print(grad.shape)
        grad = torch.abs(grad)
        patch_grad = F.conv2d(grad, filter, stride=patch_size)
        patch_grad = patch_grad.view(patch_grad.size(0), -1)
        max_patch_index = patch_grad.argsort(descending=True)[:, :args.num_patch]
    elif args.patch_select == 'Attn':
        '''attention based method'''
        atten_layer = atten[args.atten_select].mean(dim=1)
        atten_layer = atten_layer.mean(dim=-2)[:, 1:]
        max_patch_index = atten_layer.argsort(descending=True)[:, :args.num_patch]
    elif args.patch_select == 'Fixed':
        max_patch_index = torch.zeros((X.size(0), args.num_patch))
    else:
        print(f'Unknown patch_select: {args.patch_select}')
        raise
    '''build mask'''
    mask = torch.zeros([X.size(0), 1, X.size(2), X.size(3)]).cuda()
    if args.sparse_pixel_num != 0:
        learnable_mask = mask.clone()

    for j in range(X.size(0)):
        index_list = max_patch_index[j]
        for index in index_list:
            if args.patch_select == 'Fixed':
                index = 0
            row = (index // patch_num_per_line) * patch_size
            column = (index % patch_num_per_line) * patch_size

            if args.sparse_pixel_num != 0:
                learnable_mask.data[j, :, row:row + patch_size, column:column + patch_size] = torch.rand(
                    [patch_size, patch_size])
            mask[j, :, row:row + patch_size, column:column + patch_size] = 1

    '''adv attack'''
    max_patch_index_matrix = max_patch_index[:, 0]
    max_patch_index_matrix = max_patch_index_matrix.repeat(197, 1)
    max_patch_index_matrix = max_patch_index_matrix.permute(1, 0)
    max_patch_index_matrix = max_patch_index_matrix.flatten().long()

    if args.mild_l_inf == 0:
        '''random init delta'''
        delta = (torch.rand_like(X) - mu) / std
    else:
        '''constrain delta: range [x-epsilon, x+epsilon]'''
        epsilon = args.mild_l_inf / std
        delta = 2 * epsilon * torch.rand_like(X) - epsilon + X

    delta.data = clamp(delta, (0 - mu) / std, (1 - mu) / std)
    original_img = X.clone()
    if args.random_sparse_pixel:
        '''random select pixels'''
        sparse_mask = torch.zeros_like(mask)
        learnable_mask_temp = learnable_mask.view(learnable_mask.size(0), -1)
        sparse_mask_temp = sparse_mask.view(sparse_mask.size(0), -1)
        value, _ = learnable_mask_temp.sort(descending=True)
        threshold = value[:, args.sparse_pixel_num - 1].view(-1, 1)
        sparse_mask_temp[learnable_mask_temp >= threshold] = 1
        mask = sparse_mask

    if args.sparse_pixel_num == 0 or args.random_sparse_pixel:
        X = torch.mul(X, 1 - mask)
    else:
        '''select by learnable mask'''
        learnable_mask.requires_grad = True
    delta = delta.cuda()
    delta.requires_grad = True

    opt = torch.optim.Adam([delta], lr=args.attack_learning_rate)
    if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
        mask_opt = torch.optim.Adam([learnable_mask], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)

    '''Start Adv Attack'''
    for train_iter_num in range(args.train_attack_iters):
        model.zero_grad()
        opt.zero_grad()

        '''Build Sparse Patch attack binary mask'''
        if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
            if train_iter_num < args.learnable_mask_stop:
                mask_opt.zero_grad()
                sparse_mask = torch.zeros_like(mask)
                learnable_mask_temp = learnable_mask.view(learnable_mask.size(0), -1)
                sparse_mask_temp = sparse_mask.view(sparse_mask.size(0), -1)
                value, _ = learnable_mask_temp.sort(descending=True)

                threshold = value[:, args.sparse_pixel_num-1].view(-1, 1)
                sparse_mask_temp[learnable_mask_temp >= threshold] = 1

                '''inference as sparse_mask but backward as learnable_mask'''
                temp_mask = ((sparse_mask - learnable_mask).detach() + learnable_mask) * mask
            else:
                temp_mask = sparse_mask

            X = original_img * (1-sparse_mask)        
            out, atten = model(X + torch.mul(delta, temp_mask))
 
        else:
            out, atten = model(X + torch.mul(delta, mask))

        criterion = nn.CrossEntropyLoss().cuda()
        
        '''final CE-loss'''
        target_y = torch.zeros_like(y).cuda()
        loss = criterion(out, target_y)
        # loss = criterion(out, y)

        if args.attack_mode == 'Attention':
            grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
            ce_loss_grad_temp = grad.view(X.size(0), -1).detach().clone()
            if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                mask_grad = torch.autograd.grad(loss, learnable_mask, retain_graph=True)[0]

            # Attack the first 6 layers' Attn
            range_list = range(len(atten)//2)
            for atten_num in range_list:
                if atten_num == 0:
                    continue
                atten_map = atten[atten_num]
                atten_map = atten_map.mean(dim=1)
                atten_map = atten_map.view(-1, atten_map.size(-1))
                atten_map = -torch.log(atten_map)

                atten_loss = F.nll_loss(atten_map, max_patch_index_matrix + 1)
                atten_grad = torch.autograd.grad(atten_loss, delta, retain_graph=True)[0]

                atten_grad_temp = atten_grad.view(X.size(0), -1)
                cos_sim = F.cosine_similarity(atten_grad_temp, ce_loss_grad_temp, dim=1)

                if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                    mask_atten_grad = torch.autograd.grad(atten_loss, learnable_mask, retain_graph=True)[0]

                '''PCGrad'''
                atten_grad = PCGrad(atten_grad_temp, ce_loss_grad_temp, cos_sim, grad.shape)
                if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
                    mask_atten_grad_temp = mask_atten_grad.view(mask_atten_grad.size(0), -1)
                    ce_mask_grad_temp = mask_grad.view(mask_grad.size(0), -1)
                    mask_cos_sim = F.cosine_similarity(mask_atten_grad_temp, ce_mask_grad_temp, dim=1)
                    mask_atten_grad = PCGrad(mask_atten_grad_temp, ce_mask_grad_temp, mask_cos_sim, mask_atten_grad.shape)

                grad += atten_grad * args.atten_loss_weight
                if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
                    mask_grad += mask_atten_grad * args.atten_loss_weight
        else:
            '''no attention loss'''
            if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
                mask_grad = torch.autograd.grad(loss, learnable_mask)[0]
            else:
                grad = torch.autograd.grad(loss, delta)[0]

        opt.zero_grad()
        delta.grad = -grad
        opt.step()
        scheduler.step()

        if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
            mask_opt.zero_grad()
            learnable_mask.grad = -mask_grad
            mask_opt.step()

            learnable_mask_temp = learnable_mask.view(X.size(0), -1)
            learnable_mask.data -= learnable_mask_temp.min(-1)[0].view(-1, 1, 1, 1)
            learnable_mask.data += 1e-6
            learnable_mask.data *= mask

        '''l2 constrain'''
        if args.mild_l_2 != 0:
            radius = (args.mild_l_2 / std).squeeze()
            perturbation = (delta.detach() - original_img) * mask
            l2 = torch.linalg.norm(perturbation.view(perturbation.size(0), perturbation.size(1), -1), dim=-1)
            radius = radius.repeat([l2.size(0), 1])
            l2_constraint = radius / l2
            l2_constraint[l2 < radius] = 1.
            l2_constraint = l2_constraint.view(l2_constraint.size(0), l2_constraint.size(1), 1, 1)
            delta.data = original_img + perturbation * l2_constraint

        '''l_inf constrain'''
        if args.mild_l_inf != 0:
            epsilon = args.mild_l_inf / std
            delta.data = clamp(delta, original_img - epsilon, original_img + epsilon)

        delta.data = clamp(delta, (0 - mu) / std, (1 - mu) / std)
    
    perturb_x = X + torch.mul(delta, mask)

    return perturb_x

def patch_fool_fixed(model, X, y, my_index, args):
    patch_size = 16    
    filter = torch.ones([1, 3, patch_size, patch_size]).float().cuda()
    mu = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1).cuda()
    std = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1).cuda()

    patch_num_per_line = int(X.size(-1) / patch_size)
    delta = torch.zeros_like(X).cuda()
    delta.requires_grad = True
    '''choose patch'''
    # # max_patch_index size: [Batch, num_patch attack]
    # if args.patch_select == 'Rand':
    #     '''random choose patch'''
    #     max_patch_index = np.random.randint(0, 14 * 14, (X.size(0), args.num_patch))
    #     max_patch_index = torch.from_numpy(max_patch_index)
    # elif args.patch_select == 'Saliency':
    #     '''gradient based method'''
    #     grad = torch.autograd.grad(loss, delta)[0]
    #     # print(grad.shape)
    #     grad = torch.abs(grad)
    #     patch_grad = F.conv2d(grad, filter, stride=patch_size)
    #     patch_grad = patch_grad.view(patch_grad.size(0), -1)
    #     max_patch_index = patch_grad.argsort(descending=True)[:, :args.num_patch]
    # elif args.patch_select == 'Attn':
    #     '''attention based method'''
    #     atten_layer = atten[args.atten_select].mean(dim=1)
    #     atten_layer = atten_layer.mean(dim=-2)[:, 1:]
    #     max_patch_index = atten_layer.argsort(descending=True)[:, :args.num_patch]
    # elif args.patch_select == 'Fixed':
    #     max_patch_index = torch.zeros((X.size(0), args.num_patch))
    # else:
    #     print(f'Unknown patch_select: {args.patch_select}')
    #     raise
    max_patch_index = my_index
    '''build mask'''
    mask = torch.zeros([X.size(0), 1, X.size(2), X.size(3)]).cuda()
    if args.sparse_pixel_num != 0:
        learnable_mask = mask.clone()

    for j in range(X.size(0)):
        index_list = max_patch_index[j]
        for index in index_list:
            # if args.patch_select == 'Fixed':
            #     index = 0
            row = (index // patch_num_per_line) * patch_size
            column = (index % patch_num_per_line) * patch_size

            if args.sparse_pixel_num != 0:
                learnable_mask.data[j, :, row:row + patch_size, column:column + patch_size] = torch.rand(
                    [patch_size, patch_size])
            mask[j, :, row:row + patch_size, column:column + patch_size] = 1

    '''adv attack'''
    max_patch_index_matrix = max_patch_index[:, 0]
    max_patch_index_matrix = max_patch_index_matrix.repeat(197, 1)
    max_patch_index_matrix = max_patch_index_matrix.permute(1, 0)
    max_patch_index_matrix = max_patch_index_matrix.flatten().long()

    if args.mild_l_inf == 0:
        '''random init delta'''
        delta = (torch.rand_like(X) - mu) / std
    else:
        '''constrain delta: range [x-epsilon, x+epsilon]'''
        epsilon = args.mild_l_inf / std
        delta = 2 * epsilon * torch.rand_like(X) - epsilon + X

    delta.data = clamp(delta, (0 - mu) / std, (1 - mu) / std)
    original_img = X.clone()
    if args.random_sparse_pixel:
        '''random select pixels'''
        sparse_mask = torch.zeros_like(mask)
        learnable_mask_temp = learnable_mask.view(learnable_mask.size(0), -1)
        sparse_mask_temp = sparse_mask.view(sparse_mask.size(0), -1)
        value, _ = learnable_mask_temp.sort(descending=True)
        threshold = value[:, args.sparse_pixel_num - 1].view(-1, 1)
        sparse_mask_temp[learnable_mask_temp >= threshold] = 1
        mask = sparse_mask

    if args.sparse_pixel_num == 0 or args.random_sparse_pixel:
        X = torch.mul(X, 1 - mask)
    else:
        '''select by learnable mask'''
        learnable_mask.requires_grad = True
    delta = delta.cuda()
    delta.requires_grad = True

    opt = torch.optim.Adam([delta], lr=args.attack_learning_rate)
    if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
        mask_opt = torch.optim.Adam([learnable_mask], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)

    '''Start Adv Attack'''
    for train_iter_num in range(args.train_attack_iters):
        model.zero_grad()
        opt.zero_grad()

        '''Build Sparse Patch attack binary mask'''
        if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
            if train_iter_num < args.learnable_mask_stop:
                mask_opt.zero_grad()
                sparse_mask = torch.zeros_like(mask)
                learnable_mask_temp = learnable_mask.view(learnable_mask.size(0), -1)
                sparse_mask_temp = sparse_mask.view(sparse_mask.size(0), -1)
                value, _ = learnable_mask_temp.sort(descending=True)

                threshold = value[:, args.sparse_pixel_num-1].view(-1, 1)
                sparse_mask_temp[learnable_mask_temp >= threshold] = 1

                '''inference as sparse_mask but backward as learnable_mask'''
                temp_mask = ((sparse_mask - learnable_mask).detach() + learnable_mask) * mask
            else:
                temp_mask = sparse_mask

            X = original_img * (1-sparse_mask)        
            out, atten = model(X + torch.mul(delta, temp_mask))
 
        else:
            out, atten = model(X + torch.mul(delta, mask))

        criterion = nn.CrossEntropyLoss().cuda()
        
        '''final CE-loss'''
        # target_y = torch.zeros_like(y).cuda()
        # loss = criterion(out, target_y)
        loss = criterion(out, y)

        if args.attack_mode == 'Attention':
            grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
            ce_loss_grad_temp = grad.view(X.size(0), -1).detach().clone()
            if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                mask_grad = torch.autograd.grad(loss, learnable_mask, retain_graph=True)[0]

            # Attack the first 6 layers' Attn
            range_list = range(len(atten)//2)
            for atten_num in range_list:
                if atten_num == 0:
                    continue
                atten_map = atten[atten_num]
                atten_map = atten_map.mean(dim=1)
                atten_map = atten_map.view(-1, atten_map.size(-1))
                atten_map = -torch.log(atten_map)

                atten_loss = F.nll_loss(atten_map, max_patch_index_matrix + 1)
                atten_grad = torch.autograd.grad(atten_loss, delta, retain_graph=True)[0]

                atten_grad_temp = atten_grad.view(X.size(0), -1)
                cos_sim = F.cosine_similarity(atten_grad_temp, ce_loss_grad_temp, dim=1)

                if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                    mask_atten_grad = torch.autograd.grad(atten_loss, learnable_mask, retain_graph=True)[0]

                '''PCGrad'''
                atten_grad = PCGrad(atten_grad_temp, ce_loss_grad_temp, cos_sim, grad.shape)
                if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
                    mask_atten_grad_temp = mask_atten_grad.view(mask_atten_grad.size(0), -1)
                    ce_mask_grad_temp = mask_grad.view(mask_grad.size(0), -1)
                    mask_cos_sim = F.cosine_similarity(mask_atten_grad_temp, ce_mask_grad_temp, dim=1)
                    mask_atten_grad = PCGrad(mask_atten_grad_temp, ce_mask_grad_temp, mask_cos_sim, mask_atten_grad.shape)

                grad += atten_grad * args.atten_loss_weight
                if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
                    mask_grad += mask_atten_grad * args.atten_loss_weight
        else:
            '''no attention loss'''
            if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
                mask_grad = torch.autograd.grad(loss, learnable_mask)[0]
            else:
                grad = torch.autograd.grad(loss, delta)[0]

        opt.zero_grad()
        opt.step()
        scheduler.step()

        if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
            mask_opt.zero_grad()
            learnable_mask.grad = -mask_grad
            mask_opt.step()

            learnable_mask_temp = learnable_mask.view(X.size(0), -1)
            learnable_mask.data -= learnable_mask_temp.min(-1)[0].view(-1, 1, 1, 1)
            learnable_mask.data += 1e-6
            learnable_mask.data *= mask

        '''l2 constrain'''
        if args.mild_l_2 != 0:
            radius = (args.mild_l_2 / std).squeeze()
            perturbation = (delta.detach() - original_img) * mask
            l2 = torch.linalg.norm(perturbation.view(perturbation.size(0), perturbation.size(1), -1), dim=-1)
            radius = radius.repeat([l2.size(0), 1])
            l2_constraint = radius / l2
            l2_constraint[l2 < radius] = 1.
            l2_constraint = l2_constraint.view(l2_constraint.size(0), l2_constraint.size(1), 1, 1)
            delta.data = original_img + perturbation * l2_constraint

        '''l_inf constrain'''
        if args.mild_l_inf != 0:
            epsilon = args.mild_l_inf / std
            delta.data = clamp(delta, original_img - epsilon, original_img + epsilon)

        delta.data = clamp(delta, (0 - mu) / std, (1 - mu) / std)

    perturb_x = X + torch.mul(delta, mask)

    return perturb_x

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # import ipdb
    # ipdb.set_trace()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
