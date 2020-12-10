'''
A landmark-driven method on Facial Expression Recognition (FER) on FER2013 Dataset
(https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
'''

from datagen import TrainSet, ValidationSet, TestSet

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import time
from models import *

class ExpRecognition():

    def prepare_devices(self, gpu_ids, landmark_num=68, image_width=48):
        str_ids = gpu_ids.split(',')
        self.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.gpu_ids.append(id)
        if len(self.gpu_ids) > 0:
            torch.cuda.set_device(self.gpu_ids[0])
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.landmark_num = landmark_num
        self.image_width = image_width

    def load_train_data(self, image_path, label_path, batch_size=128, num_workers=4):
        trainset = TrainSet(image_path, label_path, self.landmark_num, self.image_width)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        self.mean_landmark, valid_landmark_num = trainset.cal_mean_landmark()
        print('mean landmark:')
        print(self.mean_landmark)
        print('valid landmark number:')
        print(int(valid_landmark_num))
        self.mean_landmark = torch.FloatTensor(self.mean_landmark).to(self.device)
    
    def load_validation_data(self, image_path, label_path, num_workers=4):
        validationset = ValidationSet(image_path, label_path)
        self.validation_loader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False, num_workers=num_workers)

    def load_test_data(self, image_path, num_workers=4):
        testset = TestSet(image_path)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_workers)

    def prepare_tool(self, start_lr = 1e-2, learning_rate_decay_start = 100, total_epoch = 3000, model_path = None, \
                        beta = 0.7, margin_1 = 0.5, margin_2 = 0.4, relabel_epoch = 1800):
        # model
        # self.model = VGG('VGG19', landmark_num=self.landmark_num) # use VGG model
        self.model = ResNet18(landmark_num=self.landmark_num) # use ResNet18 model
        if model_path is not None:
            assert(torch.cuda.is_available())
            self.model.to(self.device)
            self.model = nn.DataParallel(self.model, self.gpu_ids)
            ck = torch.load(model_path)
            self.model.load_state_dict(ck['net'])
        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.model.to(self.device)
            self.model = nn.DataParallel(self.model, self.gpu_ids)
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=start_lr)
        # loss function
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

        # load related setting
        self.beta = beta
        self.margin_1 = margin_1
        self.margin_2 = margin_2
        self.relabel_epoch = relabel_epoch

        # record messages
        self.start_lr = start_lr
        self.learning_rate_decay_start = max(0, learning_rate_decay_start)
        self.total_epoch = total_epoch

    def train(self, epoch):
        start = time.time()
        self.model.train()
        total_rr_loss = 0.0
        total_ce_loss = 0.0
        total_lm_loss = 0.0
        total_loss = 0.0
        total_num = 0
        for batch_idx, (img, label, landmark, have_landmark, index) in enumerate(self.train_loader):
            img, label, landmark, have_landmark = \
                img.to(self.device).float(), label.to(self.device).long(), \
                    landmark.to(self.device).float(), have_landmark.to(self.device).long()

            # Self-attention Importance Weighting Module
            attention_weights, weighted_prob, land_2d = self.model(img)

            '''SCN module in PyTorch.
            Reference:
            [1] Kai Wang, Xiaojiang Peng, Jianfei Yang, Shijian Lu, Yu Qiao
                Suppressing Uncertainties for Large-Scale Facial Expression Recognition. arXiv:2002.10392
            '''

            # Rank Regularization Module
            batch_size = img.shape[0]
            tops = int(batch_size * self.beta)
            _, top_idx = torch.topk(attention_weights.squeeze(), tops)
            _, down_idx = torch.topk(attention_weights.squeeze(), batch_size - tops, largest=False)
            high_group = attention_weights[top_idx]
            low_group = attention_weights[down_idx]
            high_mean = torch.mean(high_group)
            low_mean = torch.mean(low_group)
            diff = low_mean - high_mean + self.margin_1

            # Rank Regularization Loss
            if diff > 0.0:
                RR_loss = diff
            else:
                RR_loss = 0.0

            # Cross Entropy Loss
            CE_loss = self.loss_fn(weighted_prob, label)

            # Landmark Loss
            land_2d += self.mean_landmark
            LM_loss = torch.mean(torch.abs(land_2d-landmark) * have_landmark[:,:,None])

            # Whole Loss
            # factor = 1.0 * (self.total_epoch - epoch) / self.total_epoch
            loss = RR_loss + CE_loss + LM_loss

            if epoch >= self.learning_rate_decay_start:
                lr = self.start_lr * (self.total_epoch - epoch) / (self.total_epoch - self.learning_rate_decay_start)
                self.set_lr(self.optimizer, lr)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_rr_loss += RR_loss * batch_size
            total_ce_loss += CE_loss.item() * batch_size
            total_lm_loss += LM_loss.item() * batch_size
            total_loss += loss.item() * batch_size
            total_num += batch_size

            # Relabeling Module
            if epoch >= self.relabel_epoch:
                sm_prob = torch.softmax(weighted_prob, dim=1)
                prob_max, predicted_labels = torch.max(sm_prob, 1)
                prob_gt = torch.gather(sm_prob, 1, label.view(-1,1)).squeeze()
                t_or_f = prob_max - prob_gt > self.margin_2
                update_idx = t_or_f.nonzero().squeeze()
                label_index = index[update_idx]
                relabels = predicted_labels[update_idx]
                self.train_loader.dataset.labels[label_index.cpu().numpy()] = relabels.cpu().numpy()
        end = time.time()

        print('epoch_' + str(epoch) + ':\tspend ' + str(end-start) + 's')
        print('\tloss: ' + '{:3.6f}'.format(total_loss/total_num) + \
            '\trr loss: ' + '{:3.6f}'.format(total_rr_loss/total_num) + \
                '\tce loss: ' + '{:3.6f}'.format(total_ce_loss/total_num) + \
                    '\tlm loss: ' + '{:3.6f}'.format(total_lm_loss/total_num))

    def validation(self, validation_path, epoch):
        start = time.time()
        self.model.eval()
        total_loss = 0.0
        total_num = 0
        validation_path += (str(epoch) + '.txt')
        file = open(validation_path, 'w')
        with torch.no_grad():
            for batch_idx, (img, label) in enumerate(self.validation_loader):
                img, label = img.to(self.device).float(), label.to(self.device).long()
                _, weighted_prob, _ = self.model(img)
                loss = self.loss_fn(weighted_prob, label)

                total_loss += loss.item() * img.shape[0]
                total_num += img.shape[0]

                _, predicted = torch.max(weighted_prob.data, 1)
                file.write(str(int(predicted.data)))
                file.write('\n')
        file.close()
        end = time.time()
        print('validation:\tspend ' + str(end-start) + 's')
        print('\tloss: ' + '{:3.6f}'.format(total_loss/total_num))

    def test(self, test_path, epoch):
        start = time.time()
        self.model.eval()
        test_path += (str(epoch) + '.txt')
        file = open(test_path, 'w')
        with torch.no_grad():
            for batch_idx, (img) in enumerate(self.test_loader):
                img = img.to(self.device).float()
                _, weighted_prob, _ = self.model(img)

                _, predicted = torch.max(weighted_prob.data, 1)
                file.write(str(int(predicted.data)))
                file.write('\n')
        file.close()
        end = time.time()
        print('test:\tspend ' + str(end-start) + 's')

    def save_model(self, epoch, save_path = './model_save/resnet18_'):
        state = {'net':self.model.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':epoch}
        torch.save(state, save_path+str(epoch)+'.pth')

    def set_lr(self, optimizer, lr):
        for group in optimizer.param_groups:
            group['lr'] = lr