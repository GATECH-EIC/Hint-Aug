
import utils as myutils
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualize import *
import torch

a1 = []
a2 = []
for i in range(12):
    get1 = np.load('epoch_0/attn_' + str(i) + '.npy')
    get1 = torch.from_numpy(get1).cuda()
    a1.append(get1)
    get2 = np.load('epoch_99/attn_' + str(i) + '.npy')
    # get2 = np.load('../samples/attn_' + str(i) + '.npy')
    get2 = torch.from_numpy(get2).cuda()
    a2.append(get2)
# print(torch.eq(a1[2].mean(dim=1), a2[2].mean(dim=1)))
drawPatch(a1, a2, 9, 1, grid_index=10)
# load_checkpoint(model.module, '/home/sw99/NOAH/saves/few-shot_food-101_shot-16_seed-2_lr-0.005_wd-0.0001_adapter/checkpoint.pth', strict=False)

# exit()