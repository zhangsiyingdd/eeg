
import glob
import os
from PIL import Image
import mne
import os
import numpy as np
# from dask.array.random import standard_normal
from scipy.io import loadmat
from scipy import signal
import scipy
from sklearn.discriminant_analysis import _cov
import torch
#
# import open_clip
#
#
#
# model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
# tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
#
# # # 通过预训练模型给出图片的特征
# # model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
# #     'hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
# # tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
# device = "cuda" if torch.cuda.is_available() else "cpu"
# # img_test_feature=[]
# # test_image_path='../dataset/zx'
# # dir=os.listdir(test_image_path)[:-1]
# # for img_label in dir:
# #     image = preprocess_val(
# #         Image.open(test_image_path + '\\' + img_label)).unsqueeze(0).to(device)
# #
# #     with torch.no_grad():
# #         image_fea = model.encode_image(image)
# #     img_test_feature.append(image_fea)
# image_path='../dataset/lx'
# dir = os.listdir(image_path)
# img_train_feature=[]
# img_val_feature=[]
#
# for img_label in dir:
#     for i in range(10):
#         if i <8:
#             image = preprocess_val(Image.open(image_path+'\\'+img_label+'\\{:02d}'.format(i+1)+'.jpg')).unsqueeze(0).to(device)
#
#             with torch.no_grad():
#                 image_fea = model.encode_image(image)
#             img_train_feature.append(image_fea)
#             img_train_feature.append(image_fea)
#             img_train_feature.append(image_fea)
#             img_train_feature.append(image_fea)
#             img_train_feature.append(image_fea)
#
#
#         else:
#             image = preprocess_val(Image.open(image_path + '\\' + img_label + '\\{:02d}'.format(i+1)+ '.jpg')).unsqueeze(0).to(device)
#
#             with torch.no_grad():
#                 image_fea = model.encode_image(image)
#             img_val_feature.append(image_fea)
#             img_val_feature.append(image_fea)
#             img_val_feature.append(image_fea)
#             img_val_feature.append(image_fea)
#             img_val_feature.append(image_fea)
#
#
# # #
# # stacked_tensor = torch.stack(img_test_feature)
# # stacked_tensor_m = stacked_tensor.reshape(stacked_tensor.shape[0], -1)
# # np.save('img_test_feature_h14.npy', stacked_tensor_m)
#
#
#
#
# stacked_tensor = torch.stack(img_train_feature)
# stacked_tensor_m = stacked_tensor.reshape(stacked_tensor.shape[0], -1)
# np.save('img_train_feature_h14.npy', stacked_tensor_m)
#
# stacked_tensor = torch.stack(img_val_feature)
# stacked_tensor_m = stacked_tensor.reshape(stacked_tensor.shape[0], -1)
# np.save('img_val_feature_h14.npy', stacked_tensor_m)
# print("图片完成")

#导入脑电数据
def load_data(data_bdf_name, evt_bdf_name,window_size):
    # 导入bdf文件数据

    bdf_path = data_bdf_name
    event_path = evt_bdf_name
    raw = mne.io.read_raw_bdf(bdf_path)


    raw.load_data()
    raw_event = mne.read_annotations(event_path)
    print(raw.info)
    print(raw_event)
    raw.set_annotations(raw_event, emit_warning=False)
    events, set_trigger = mne.events_from_annotations(raw)
    print(set_trigger['241'])
    picks = mne.pick_types(raw.info, eeg=True, exclude=["ECG", "HEOR", "HEOL", "VEOU", "VEOL"])
    idx_target = np.where(events[:, 2] == set_trigger['241'])[0]
    events = np.delete(events, idx_target, 0)
    idx_target = np.where(events[:, 2] == set_trigger['242'])[0]
    events = np.delete(events, idx_target, 0)
    idx_target = np.where(events[:, 2] == set_trigger['243'])[0]
    events = np.delete(events, idx_target, 0)
    idx_target = np.where(events[:, 2] == set_trigger['250'])[0]
    events = np.delete(events, idx_target, 0)
    # idx_target = np.where(events[:, 2] == set_trigger['251'])[0]
    # events = np.delete(events, idx_target, 0)
    # 读取Epoch_non_target数据64*600

    data_target=[]
    for i in range(1,161):
        epochs_img_condition= mne.Epochs(raw, events, event_id=set_trigger[str(i)], tmin=-1, tmax=1, baseline=(None,0),picks=picks)

        epochs_img_condition.load_data()
        epochs_img_condition.resample(500)
        data_target.append(epochs_img_condition.get_data())


    data_target = np.array(data_target)#160,50,59,250
    data_list=[]


    for i in range(data_target.shape[0]):

        data_temp=[]
        for j in range(10):

            data_temp.append(data_target[i, 10*j , :, :])
        data_list.append(data_temp)
    #160,10,59.250
    data_list=np.array(data_list)
    data_list=data_list[:, :, :, 500+window_size*50:500+window_size*50+250]


    return data_list



def whiten(data):#160*10*59*250

    sigma_cond=np.empty((data.shape[0],data.shape[2],data.shape[2]))
    for j in range(data.shape[0]):#leibie
        cond_data=data[j]
        sigma_cond[j] = np.mean([_cov(np.transpose(cond_data[e]),
                                          shrinkage='auto') for e in range(cond_data.shape[0])],axis=0)
    sigma_part = sigma_cond.mean(axis=0)


    sigma_tot = sigma_part
    sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)
    whited_data=np.empty((data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            whited_data[i, j] = ((data[i, j].T)@ sigma_inv).T

    return whited_data,sigma_inv

def split_data(data):#160,2,10,59,250
    train_data = []
    val_data = []


    for j in range(160):
        for i in range(10):
            if i<8:
                train_data.append(data[j,i])

            else:
                val_data.append(data[j,i])

    train_data = np.array(train_data)
    val_data = np.array(val_data)

    return train_data, val_data


def notch_filter(data,notch_HZ):
    notch_filtered_data=np.empty((data.shape[0], data.shape[1], data.shape[2],data.shape[3]))
    b, a = signal.iirnotch(notch_HZ, 40, fs=500)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):

            notch_filtered_data[i,j]= signal.lfilter(b, a, data[i,j])
    return notch_filtered_data

def bp_filter(data,samp_rate):
    f_1, f_2, fs = 3, 100, samp_rate
    wn = [f_1 * 2 / fs, f_2 * 2 / fs]
    b, a = signal.butter(3, wn, 'bandpass')
    filter_data = signal.filtfilt(b, a, data)
    return filter_data
def z_score(data):
    epsilon = 1e-8  # 防止除零

    # 1. 对每个试次的所有数据进行标准化（跨类别、通道和时间点）
    # 计算每个试次的均值和标准差，保持维度一致
    trial_means = np.mean(data, axis=( 2, 3), keepdims=True)  # 形状：(160, 10, 1, 1)
    trial_stds = np.std(data, axis=(2, 3), keepdims=True)  # 形状：(160, 10, 1, 1)
    z_score_per_trial = (data - trial_means) / (trial_stds + epsilon)
    return z_score_per_trial

def windows_img_feature(window_number):
    train_img=np.load('img_train_feature_h14.npy',allow_pickle= True)
    val_img=np.load('img_val_feature_h14.npy',allow_pickle=True)
    combined_train_img = np.concatenate([train_img, train_img,train_img,train_img], axis=0)
    combined_val_img = np.concatenate([val_img, val_img,val_img,val_img], axis=0)
    return combined_train_img,combined_val_img

bdf_path = 'Z:\个人文档\张思莹文档\\7-24马博数据\lx/data.bdf'
event_path = 'Z:\个人文档\张思莹文档\\7-24马博数据\lx/evt.bdf'

#
# bdf_path = '../..//数据/4-14-zsy-第一次/离线/data.bdf'
# event_path = '../..//数据/4-14-zsy-第一次//离线/evt.bdf'
combined_train_data = np.random.rand(0, 59, 250)
combined_val_data = np.random.rand(0, 59, 250)
train_img_w4,val_img_w4=windows_img_feature(5)
for i in range(1,5):

    eeg_data = load_data(bdf_path, event_path,i)
    eeg_data=notch_filter(eeg_data,50)
    # whited_data,white_array=whiten(eeg_data)
    standard_data=z_score(eeg_data)
    train_data,val_data=split_data(standard_data)
    combined_train_data = np.concatenate([combined_train_data, train_data], axis=0)
    combined_val_data = np.concatenate([combined_val_data, val_data], axis=0)

np.save('img_train_feature_h14_w4.npy', train_img_w4)
np.save('img_val_feature_h14_w4.npy', val_img_w4)
# np.save('white_array.npy',white_array)
np.save('eeg_train_data.npy', combined_train_data)
np.save('eeg_val_data.npy', combined_val_data)

