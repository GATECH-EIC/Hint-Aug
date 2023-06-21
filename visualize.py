import argparse
import copy
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image, ImageDraw
import numpy as np
    
mu = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def show_img(X, attack, epoch):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Origin')
    ax2.set_title('Attack')
    X = X * std.numpy() + mu.numpy()
    attack = attack * std.numpy() + mu.numpy()
    _ = ax1.imshow(X.transpose(1, 2, 0))
    _ = ax2.imshow(attack.transpose(1, 2, 0))
    fig.savefig('Epoch_' + str(epoch) + '/Compared' + '.png')
    plt.close(fig)

def grid_show(to_shows, cols):
    rows = (len(to_shows)-1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows*8.5, cols*2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.show()

def visualize_head(att_map):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()
    
def visualize_heads(att_map, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)

def gray2rgb(image):
    return np.repeat(image[...,np.newaxis],3,2)
    
def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
        
    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H/grid_size[0])
    delta_W = int(W/grid_size[1])
    
    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]
    
    padded_image = np.hstack((padding,image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W/4),int(delta_H/4)),'CLS', fill=(0,0,0)) # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask),cls_weight)
    cls_weight = cls_weight / max(np.max(mask),cls_weight)
    
    if len(padding.shape) == 3:
        padding = padding[:,:,0]
        padding[:,:] = np.min(mask)
    mask_to_pad = np.ones((1,1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H,  :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask
    
    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1],4))
    meta_mask[delta_H:,0: delta_W, :] = 1 
    
    return padded_image, padded_mask, meta_mask
    

def visualize_grid_to_grid_with_cls(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]

    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    padded_image ,padded_mask, meta_mask = cls_padding(image, mask, cls_weight, grid_size)
    
    if grid_index != 0: # adjust grid_index since we pad our image
        grid_index = grid_index + (grid_index-1) // grid_size[1]
        
    grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1]+1))
    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(grid_image)
    ax[0].axis('off')
    
    ax[1].imshow(grid_image)
    ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
    ax[1].imshow(meta_mask)
    ax[1].axis('off')

    

def visualize_grid_to_grid(att_map, grid_index, image, grid_size=14, alpha=1, name='Test'):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    H,W = att_map.shape
    with_cls_token = False
      
    grid_image = highlight_grid(image, [grid_index], grid_size)
    
    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    # mask = Image.fromarray(mask).resize((image.size))
    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(grid_image)
    ax[0].axis('off')
    
    # ax[1].imshow(grid_image)
    ax[1].imshow(mask, alpha=alpha, cmap='viridis')
    rect = patches.Rectangle((7-0.5, 7-0.5), 1, 1, edgecolor='r', linewidth=3, fc='None')
    ax[1].patch.set_alpha(1)
    ax[1].add_patch(rect)
    ax[1].axis('off') 
    fig.savefig(name + '.png')
    plt.close(fig)
    
def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a = ImageDraw.ImageDraw(image)
        a.rectangle([(y*w,x*h),(y*w+w,x*h+h)],fill =None,outline ='red',width =2)
    return image

def deltaPatch(attn1, attn2, grid_index, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    delta_l0 = attn1[0][0,0,1:,1:].cpu().detach().numpy() - attn2[0][0,0,1:,1:].cpu().detach().numpy()
    delta_l5 = attn1[5][0,0,1:,1:].cpu().detach().numpy() - attn2[5][0,0,1:,1:].cpu().detach().numpy()
    delta_l11 = attn1[11][0,0,1:,1:].cpu().detach().numpy() - attn2[11][0,0,1:,1:].cpu().detach().numpy() 
    mask_l0 = delta_l0[grid_index].reshape(grid_size[0], grid_size[1])
    mask_l5 = delta_l5[grid_index].reshape(grid_size[0], grid_size[1])
    mask_l11 = delta_l11[grid_index].reshape(grid_size[0], grid_size[1])
    vMax1 = max(delta_l0.max(), attn1[0][0,0,1:,1:].cpu().detach().numpy().max(), attn2[0][0,0,1:,1:].cpu().detach().numpy().max())
    vMax2 = max(delta_l5.max(), attn1[5][0,0,1:,1:].cpu().detach().numpy().max(), attn2[5][0,0,1:,1:].cpu().detach().numpy().max())
    vMax3 = max(delta_l11.max(), attn1[11][0,0,1:,1:].cpu().detach().numpy().max(), attn2[11][0,0,1:,1:].cpu().detach().numpy().max())
    vMin1 = max(delta_l0.min(), attn1[0][0,0,1:,1:].cpu().detach().numpy().min(), attn2[0][0,0,1:,1:].cpu().detach().numpy().min())
    vMin2 = max(delta_l5.min(), attn1[5][0,0,1:,1:].cpu().detach().numpy().min(), attn2[5][0,0,1:,1:].cpu().detach().numpy().min())
    vMin3 = max(delta_l11.min(), attn1[11][0,0,1:,1:].cpu().detach().numpy().min(), attn2[11][0,0,1:,1:].cpu().detach().numpy().min())
    fig, ax = plt.subplots(1, 3, figsize=(10,7))
    ax[0].imshow(mask_l0, alpha=1, cmap='viridis', vmax=vMax1, vmin=vMin1)
    rect = patches.Rectangle((7-0.5, 7-0.5), 1, 1, edgecolor='r', linewidth=2, fc='None')
    ax[0].add_patch(rect)
    ax[0].axis('off') 
    
    ax[1].imshow(mask_l5, alpha=1, cmap='viridis', vmax=vMax2, vmin=vMin2)
    rect = patches.Rectangle((7-0.5, 7-0.5), 1, 1, edgecolor='r', linewidth=2, fc='None')
    ax[1].add_patch(rect)
    ax[1].axis('off') 

    ax[2].imshow(mask_l11, alpha=1, cmap='viridis', vmax=vMax3, vmin=vMin3)
    rect = patches.Rectangle((7-0.5, 7-0.5), 1, 1, edgecolor='r', linewidth=2, fc='None')
    ax[2].add_patch(rect)
    ax[2].axis('off') 

    fig.savefig('Attn_delta.png')
    plt.close(fig)

def drawAttn(x, attn, index, attack=False):
    x = (x * std + mu)
    x = x.permute(1, 2, 0).numpy()
    img = Image.fromarray(np.uint8(x*255))
    if attack:
        visualize_grid_to_grid(attn[0][0,0,1:,1:].cpu().detach().numpy(), index, img, name='Attack_Layer0_head0')
        visualize_grid_to_grid(attn[5][0,0,1:,1:].cpu().detach().numpy(), index, img, name='Attack_Layer5_head0')
        visualize_grid_to_grid(attn[11][0,0,1:,1:].cpu().detach().numpy(), index, img, name='Attack_Layer11_head0')
    else:
        visualize_grid_to_grid(attn[0][0,0,1:,1:].cpu().detach().numpy(), index, img, name='Layer0_head0')
        visualize_grid_to_grid(attn[5][0,0,1:,1:].cpu().detach().numpy(), index, img, name='Layer5_head0')
        visualize_grid_to_grid(attn[11][0,0,1:,1:].cpu().detach().numpy(), index, img, name='Layer11_head0')

def drawPatch(attn, attn2, layer, head, epoch, grid_index=105):
    pre_attn = attn[layer].mean(dim=1)
    pre_attn = pre_attn[0, 1:, 1:].cpu().detach().numpy()

    attack_attn = attn2[layer].mean(dim=1)
    attack_attn = attack_attn[0, 1:, 1:].cpu().detach().numpy()
    delta_attn = np.abs(pre_attn - attack_attn)
    vMax = pre_attn.max()
    vMin = pre_attn.min()

    loc_x = grid_index % 14
    loc_y = grid_index // 14

    pre_patch = pre_attn[grid_index].reshape(14, 14)
    delta_patch = delta_attn[grid_index].reshape(14, 14)
    attack_patch = attack_attn[grid_index].reshape(14, 14)
    print(np.array_equal(pre_patch, attack_patch))
    print(np.sum(np.absolute(pre_patch - attack_patch)))
    # exit()
    fig, ax = plt.subplots(1, 3, figsize=(10,7))
    ax[0].imshow(pre_patch, alpha=1, cmap='viridis')
    rect = patches.Rectangle((loc_x-0.5, loc_y-0.5), 1, 1, edgecolor='r', linewidth=2, fc='None')
    # ax[0].add_patch(rect)
    ax[0].axis('off') 
    
    ax[1].imshow(delta_patch, alpha=1, cmap='viridis', vmin=0, vmax=1)
    rect = patches.Rectangle((loc_x-0.5, loc_y-0.5), 1, 1, edgecolor='r', linewidth=0.5, fc='None')
    # ax[1].add_patch(rect)
    ax[1].axis('off')   

    ax[2].imshow(attack_patch, alpha=1, cmap='viridis')
    rect = patches.Rectangle((loc_x-0.5, loc_y-0.5), 1, 1, edgecolor='r', linewidth=2, fc='None')
    # ax[2].add_patch(rect)
    ax[2].axis('off') 

    fig.savefig('Epoch_' + str(epoch) + '/Attn_'+ str(layer) + '_Head_' + str(head) +'.png')
    plt.close(fig)

def drawPatch_np(attn, attn2, layer, head, grid_index=105):
    pre_attn = attn[layer].mean(dim=1)
    pre_attn = pre_attn[0, 1:, 1:]

    attack_attn = attn2[layer].mean(dim=1)
    attack_attn = attack_attn[0, 1:, 1:]
    delta_attn = np.abs(pre_attn - attack_attn)
    vMax = pre_attn.max()
    vMin = pre_attn.min()

    loc_x = grid_index % 14
    loc_y = grid_index // 14

    pre_patch = pre_attn[grid_index].reshape(14, 14)
    delta_patch = delta_attn[grid_index].reshape(14, 14)
    attack_patch = attack_attn[grid_index].reshape(14, 14)

    fig, ax = plt.subplots(1, 3, figsize=(10,7))
    ax[0].imshow(pre_patch, alpha=1, cmap='viridis')
    rect = patches.Rectangle((loc_x-0.5, loc_y-0.5), 1, 1, edgecolor='r', linewidth=2, fc='None')
    ax[0].add_patch(rect)
    ax[0].axis('off') 
    
    ax[1].imshow(delta_patch, alpha=1, cmap='viridis', vmin=0, vmax=1)
    rect = patches.Rectangle((loc_x-0.5, loc_y-0.5), 1, 1, edgecolor='r', linewidth=0.5, fc='None')
    ax[1].add_patch(rect)
    ax[1].axis('off')   

    ax[2].imshow(attack_patch, alpha=1, cmap='viridis')
    rect = patches.Rectangle((loc_x-0.5, loc_y-0.5), 1, 1, edgecolor='r', linewidth=2, fc='None')
    ax[2].add_patch(rect)
    ax[2].axis('off') 

    fig.savefig('Attn_'+ str(layer) + '_Head_' + str(head) +'.png')
    plt.close(fig)