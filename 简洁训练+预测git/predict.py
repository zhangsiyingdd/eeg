#导入模型文件


#导入数据

from scipy import signal
#预
import os
import argparse
import random
import itertools
import datetime
import time
import numpy as np
import pandas as pd
import mne

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from torch import Tensor

from torch.autograd import Variable
from einops.layers.torch import Rearrange

import torch
from scipy.stats import mode



class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=760, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
    def forward(self, x):
        return x
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=200):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (59, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.eegnet = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (59, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4), (1, 4)),
            nn.Dropout2d(0.5)
        )

        self.projection = nn.Sequential(
            # nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            nn.Conv2d(16, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        # x = self.tsconv(x)
        x = self.eegnet(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            FlattenHead()
        )

model_idx = 'tiaocan'
Enc_eeg=Enc_eeg()

Enc_eeg.load_state_dict(torch.load('./model/'+model_idx+'Enc_eeg_cls.pth', map_location=torch.device('cpu')), strict=False)
Proj_eeg=Proj_eeg()
Proj_eeg.load_state_dict(torch.load('./model/'+model_idx+ 'Proj_eeg_cls.pth', map_location=torch.device('cpu')),strict=False)
Proj_img=Proj_img()
Proj_img.load_state_dict(torch.load('./model/'+model_idx+'Proj_img_cls.pth', map_location=torch.device('cpu')), strict=False)

Enc_eeg.eval()
Proj_eeg.eval()
Proj_img.eval()



def load_test_data(data_bdf_name, evt_bdf_name,count_num):
    # 导入bdf文件数据

    bdf_path = data_bdf_name
    event_path = evt_bdf_name
    raw = mne.io.read_raw_bdf(bdf_path)
    raw.load_data()
    raw_event = mne.read_annotations(event_path)

    raw.set_annotations(raw_event, emit_warning=False)
    events, set_trigger = mne.events_from_annotations(raw)

    picks = mne.pick_types(raw.info, eeg=True, exclude=["ECG", "HEOR", "HEOL", "VEOU", "VEOL"])
    idx_target = np.where(events[:, 2] == set_trigger['240'])[0]
    events = np.delete(events, idx_target, 0)
    idx_target = np.where(events[:, 2] == set_trigger['241'])[0]
    events = np.delete(events, idx_target, 0)
    idx_target = np.where(events[:, 2] == set_trigger['242'])[0]
    events = np.delete(events, idx_target, 0)
    idx_target = np.where(events[:, 2] == set_trigger['243'])[0]
    events = np.delete(events, idx_target, 0)
    idx_target = np.where(events[:, 2] == set_trigger['250'])[0]
    events = np.delete(events, idx_target, 0)
    idx_target = np.where(events[:, 2] == set_trigger['251'])[0]
    events = np.delete(events, idx_target, 0)
    # 读取Epoch_non_target数据64*600

    data_target=[]
    for i in range(1,81):
        epochs_img_condition= mne.Epochs(raw, events, event_id=set_trigger[str(i)], tmin=-1, tmax=1, baseline=(None,0),picks=picks)

        epochs_img_condition.load_data()
        epochs_img_condition.resample(500)
        data_target.append(epochs_img_condition.get_data())


    data_target = np.array(data_target)



    return data_target[:,0,:,500+count_num*50:500+count_num*50+250]

def whiten(data):
    sigma_inv=np.load('white_array.npy',allow_pickle=True)
    whited_data=np.empty((data.shape[0], data.shape[1], data.shape[2]))
    for i in range(data.shape[0]):

        whited_data[i] = ((data[i].T)@ sigma_inv).T

    return whited_data
def notch_filter(data,notch_HZ):

    notch_filtered_data = np.empty((data.shape[0], data.shape[1], data.shape[2]))
    b, a = signal.iirnotch(notch_HZ, 40, fs=500)
    for i in range(data.shape[0]):

        notch_filtered_data[i] = signal.lfilter(b, a, data[i])
    return notch_filtered_data


def bp_filter(data,samp_rate):
    f_1, f_2, fs = 3, 100, samp_rate
    wn = [f_1 * 2 / fs, f_2 * 2 / fs]
    b, a = signal.butter(3, wn, 'bandpass')
    filter_data = signal.filtfilt(b, a, data)
    return filter_data

path=os.path.dirname(os.getcwd())
bdf_path = os.path.join(path, '7-24马博数据\zx', 'data.bdf')
event_path = os.path.join(path, '7-24马博数据\zx', 'evt.bdf')

# 进行循环
predict_list = []
# teeg=bp_filter(teeg,250)
for i in range(5):
    teeg = load_test_data(bdf_path, event_path,i)
    print(teeg.shape)# 80*59*250/
    teeg=notch_filter(teeg,50)
    # teeg=notch_filter(teeg,5)
    teeg=whiten(teeg)
    teeg=np.expand_dims(teeg,axis=1)
    teeg=teeg.astype(np.float32)
    # teeg = np.random.rand(10, 1 , 60, 250).astype(np.float32)
    teeg_tensor = torch.from_numpy(teeg)
    all_center=np.load('img_test_feature_h14.npy',allow_pickle=True)
    # all_center=np.random.rand(10, 512).astype(np.float32)
    all_center = torch.from_numpy(all_center)
    test_label = np.arange(80)
    tlabel=torch.from_numpy(test_label)
    total = 0
    top1 = 0
    top3 = 0
    top5 = 0
    # 进行推理



    with torch.no_grad():
        tfea =Proj_eeg(Enc_eeg(teeg_tensor))
        tfea = tfea / tfea.norm(dim=1, keepdim=True)
        similarity = (100.0 * tfea @ all_center.t()).softmax(dim=-1)  # 预测的相似度矩阵

        _, indices = similarity.topk(5)
        tt_label = tlabel.view(-1, 1)
        total += tlabel.size(0)
        top1 += (tt_label == indices[:, :1]).sum().item()
        top3 += (tt_label == indices[:, :3]).sum().item()
        top5 += (tt_label == indices).sum().item()


        top1_acc = float(top1) / float(total)
        top3_acc = float(top3) / float(total)
        top5_acc = float(top5) / float(total)
        predict_list.append(indices[:, 0])


predict_list=np.array(predict_list)
predict_category_list=np.trunc(predict_list/20)
predict_object_list=np.trunc(predict_list/2)

# final_predict_list=weight_category_choose(predict_category_list)

object_right_number=0
category_right_number=0
for i in range(80):

    label=int(i/20)
    # 结果的第一个元素是众数，第二个是出现次数
    modes, counts = mode(predict_category_list[2,:], axis=0)

    # 转换为一维数组（去除多余的维度）
    pre = predict_category_list[1,:].flatten()
    if pre[i]==label:
        category_right_number +=1
    label = int(i / 2)
    # 结果的第一个元素是众数，第二个是出现次数
    modes, counts = mode(predict_object_list[1,:], axis=0)

    # 转换为一维数组（去除多余的维度）
    pre=predict_object_list[1,:].flatten()

    if pre[i]  == label:
        object_right_number += 1

print("category_accuracy:%.6f" % (category_right_number/80))
print("object_accuracy:%.6f" % (object_right_number/80))
print('The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_acc, top3_acc, top5_acc))
