

"""
Object recognition Things-EEG2 dataset

use 250 Hz data
"""
import numpy as np

import os
import argparse
import random
import itertools
import datetime
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from torch.autograd import Variable
from einops.layers.torch import Rearrange



result_path = './results/'
model_idx = 'tiaocan'

parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with CLIP encoder')
parser.add_argument('--dnn', default='clip', type=str)
parser.add_argument('--epoch', default='200', type=int)
parser.add_argument('--num_sub', default=1, type=int,
                    help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=500, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=2002, type=int,
                    help='seed for initializing training. ')


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


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

    # Image2EEG


class IE():
    def __init__(self, args, nsub):
        super(IE, self).__init__()
        self.args = args

        self.batch_size = args.batch_size
        self.batch_size_test = 10
        self.batch_size_img = 10
        self.n_epochs = args.epoch

        self.lambda_cen = 0.003
        self.alpha = 0.5

        self.proj_dim = 256

        self.lr = 0.0008
        self.b1 = 0.5
        self.b2 = 0.999

        self.nSub = nsub

        self.start_epoch = 0



        self.pretrain = False

        self.log_write = open(result_path + "log_subject%d.txt" % self.nSub, "w")

        self.Tensor = torch.FloatTensor
        self.LongTensor = torch.LongTensor

        self.criterion_l1 = torch.nn.L1Loss()
        self.criterion_l2 = torch.nn.MSELoss()
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.Enc_eeg = Enc_eeg()
        self.Proj_eeg = Proj_eeg()
        self.Proj_img = Proj_img()
        # self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        # self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        # self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.centers = {}
        print('initial define done.')

    def get_eeg_data(self):

        train_label = []
        train_data = np.load('eeg_train_data.npy',allow_pickle=True)
        # train_data = np.mean(train_data, axis=1)
        train_data = np.expand_dims(train_data, axis=1)

        val_data = np.load('eeg_val_data.npy',allow_pickle=True)
        # val_data = np.mean(val_data, axis=1)
        val_data = np.expand_dims(val_data, axis=1)


        return train_data, train_label,val_data

    def get_image_data(self):
        train_img_feature = np.load('img_train_feature_h14_w4.npy',allow_pickle=True)

        val_img_feature = np.load('img_val_feature_h14_w4.npy',allow_pickle=True)

        return train_img_feature, val_img_feature

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):

        self.Enc_eeg.apply(weights_init_normal)
        self.Proj_eeg.apply(weights_init_normal)
        self.Proj_img.apply(weights_init_normal)
        train_img_feature, val_img_feature = self.get_image_data()

        train_eeg, _,val_eeg= self.get_eeg_data()

        # test_center = np.load(self.test_center_path + 'center_' + self.args.dnn + '.npy', allow_pickle=True)

        # shuffle the training data
        train_shuffle = np.random.permutation(len(train_eeg))
        train_eeg = train_eeg[train_shuffle]
        train_img_feature = train_img_feature[train_shuffle]

        val_shuffle = np.random.permutation(len(val_eeg))
        val_eeg = val_eeg[val_shuffle]
        val_img_feature = val_img_feature[val_shuffle]



        val_eeg = torch.from_numpy(val_eeg)
        val_image = torch.from_numpy(val_img_feature)

        train_eeg = torch.from_numpy(train_eeg)
        train_image = torch.from_numpy(train_img_feature)

        dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size,
                                                          shuffle=False)

        # test_eeg = torch.from_numpy(test_eeg)
        # # test_img_feature = torch.from_numpy(test_img_feature)
        # test_center = torch.from_numpy(test_center)
        # test_label = torch.from_numpy(test_label)
        # test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        # self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test,
        #                                                    shuffle=False)

        # Optimizers
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.Enc_eeg.parameters(), self.Proj_eeg.parameters(), self.Proj_img.parameters()),
            lr=self.lr, betas=(self.b1, self.b2))

        num = 0
        best_loss_val = np.inf

        for e in range(self.n_epochs):
            in_epoch = time.time()

            self.Enc_eeg.train()
            self.Proj_eeg.train()
            self.Proj_img.train()

            # starttime_epoch = datetime.datetime.now()

            for i, (eeg, img) in enumerate(self.dataloader):
                eeg = Variable(eeg.type(self.Tensor))
                # img = Variable(img.cuda().type(self.Tensor))
                img_features = Variable(img.type(self.Tensor))
                # label = Variable(label.cuda().type(self.LongTensor))
                labels = torch.arange(eeg.shape[0])  # used for the loss
                labels = Variable(labels.type(self.LongTensor))

                # obtain the features
                eeg_features = self.Enc_eeg(eeg)
                # img_features = self.Enc_img(img).last_hidden_state[:,0,:]

                # project the features to a multimodal embedding space
                eeg_features = self.Proj_eeg(eeg_features)
                img_features = self.Proj_img(img_features)

                # normalize the features
                eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
                img_features = img_features / img_features.norm(dim=1, keepdim=True)

                # cosine similarity as the logits
                logit_scale = self.logit_scale.exp()
                logits_per_eeg = logit_scale * eeg_features @ img_features.t()
                logits_per_img = logits_per_eeg.t()

                loss_eeg = self.criterion_cls(logits_per_eeg, labels)
                loss_img = self.criterion_cls(logits_per_img, labels)

                loss_cos = (loss_eeg + loss_img) / 2

                # total loss
                loss = loss_cos

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (e + 1) % 1 == 0:
                self.Enc_eeg.eval()
                self.Proj_eeg.eval()
                self.Proj_img.eval()
                with torch.no_grad():
                    # * validation part
                    for i, (veeg, vimg) in enumerate(self.val_dataloader):

                        veeg = Variable(veeg.type(self.Tensor))
                        vimg_features = Variable(vimg.type(self.Tensor))
                        vlabels = torch.arange(veeg.shape[0])
                        vlabels = Variable(vlabels.type(self.LongTensor))

                        veeg_features = self.Enc_eeg(veeg)
                        veeg_features = self.Proj_eeg(veeg_features)
                        vimg_features = self.Proj_img(vimg_features)

                        veeg_features = veeg_features / veeg_features.norm(dim=1, keepdim=True)
                        vimg_features = vimg_features / vimg_features.norm(dim=1, keepdim=True)

                        logit_scale = self.logit_scale.exp()
                        vlogits_per_eeg = logit_scale * veeg_features @ vimg_features.t()
                        vlogits_per_img = vlogits_per_eeg.t()

                        vloss_eeg = self.criterion_cls(vlogits_per_eeg, vlabels)
                        vloss_img = self.criterion_cls(vlogits_per_img, vlabels)

                        vloss = (vloss_eeg + vloss_img) / 2

                        if vloss <= best_loss_val:
                            best_loss_val = vloss
                            best_epoch = e

                            torch.save(self.Enc_eeg.state_dict(), './model/' + model_idx + 'Enc_eeg_cls.pth')
                            torch.save(self.Proj_eeg.state_dict(), './model/' + model_idx + 'Proj_eeg_cls.pth')
                            torch.save(self.Proj_img.state_dict(), './model/' + model_idx + 'Proj_img_cls.pth')

                print('Epoch:', e,
                      '  Cos eeg: %.4f' % loss_eeg.detach().cpu().numpy(),
                      '  Cos img: %.4f' % loss_img.detach().cpu().numpy(),
                      '  loss val: %.4f' % vloss.detach().cpu().numpy(),
                      )
                self.log_write.write('Epoch %d: Cos eeg: %.4f, Cos img: %.4f, loss val: %.4f\n' % (
                e, loss_eeg.detach().cpu().numpy(), loss_img.detach().cpu().numpy(), vloss.detach().cpu().numpy()))

        return best_epoch

def main():
    args = parser.parse_args()

    num_sub = args.num_sub
    cal_num = 0


    for i in range(num_sub):
        cal_num += 1
        starttime = datetime.datetime.now()
        # seed_n = np.random.randint(args.seed)
        seed_n = 1799
        with open("./results/seed.txt", "w") as log:
            log.write('seed is ' + str(seed_n) + '\n')
            print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        # torch.cuda.manual_seed(seed_n)
        # torch.cuda.manual_seed_all(seed_n)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        print('Subject %d' % (i + 1))
        ie = IE(args, i + 1)

        result = ie.train()

        endtime = datetime.datetime.now()
        print('best epoch is %d' % result)
        print('subject %d duration: ' % (i + 1) + str(endtime - starttime))





if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))


