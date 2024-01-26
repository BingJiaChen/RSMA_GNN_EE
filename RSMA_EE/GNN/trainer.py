import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import *
from utils import *
from model import *
import matplotlib.pyplot as plt
import time

class Trainer():
    def __init__(self,M,N,L,training_K,testing_K,batch_size,Pt=40):
        self.M = M
        self.N = N
        self.training_K = training_K
        self.testing_K = testing_K
        self.L = L
        self.b = 2
        self.Pmax = 10**((Pt-30)/10)
        self.Pc_training = 1/0.8 + 0.001*self.N*self.L
        self.Pc_testing = 1/0.8 + 0.001*self.N*self.L
        self.batch_size = batch_size
        self.dataloader = MyDataLoader(self.M,self.N,self.L,self.batch_size)
        self.device = 'cuda'
        self.model = node_update(M,N,L,3,self.Pmax,2,128).to(self.device)
        self.n_iter = 1500
        self.min_rate = 1
        self.log_interval = 10
        self.log_eval_interval = 100

    def train_batch(self):
        self.model.train()
        user_feature, H, channel_bs_user, e = self.dataloader.load_data(self.training_K,0)
        user_feature = user_feature.to(self.device)
        e = e.to(self.device)
        self.opt.zero_grad()
        W, theta, mu, e_rc = self.model(user_feature,e)
        
        loss, sum_rate, over_rate = cal_loss(W,theta,H,channel_bs_user,self.Pmax,mu,self.Pc_training,self.min_rate,e_rc)
        loss.backward()
        self.opt.step()

        return loss.item(), sum_rate.item()


    def train(self):
        self.opt = torch.optim.Adam(self.model.parameters(),lr=0.001,weight_decay=1e-6)
        train_loss = []
        train_sum_rate = []
        train_total = []
        val_sum_rate = []
        val_EE  = [1]
        val_sigma1 = []
        val_sigma2 = []
        test_sample = 3200

        for i in range(self.n_iter):
            loss, sum_rate = self.train_batch()
            train_loss.append(loss)
            train_sum_rate.append(sum_rate)
            if i% self.log_interval == 0:
                # print(f"[Train | {i}/{self.n_iter} ] loss = {np.mean(train_loss):.5f}, sum rate = {np.mean(train_sum_rate):.5f}")
                train_total.append(np.mean(train_loss))
                train_loss = []
                train_sum_rate = []

            if (i+1)%self.log_eval_interval==0 and i>=1:
                start_time = time.time()
                EE, sum_rate, over_rate = self.eval(test_sample,0)
                end_time = time.time()
                # print("execution time:",end_time-start_time)
                print(f"[Val | {i+1}/{self.n_iter} ] EE = {(EE):.5f}, sum rate = {(sum_rate):.5f}, over rate = {(over_rate):.3f}")
                val_sum_rate.append(sum_rate)
                val_EE.append(EE)
                # EE, sum_rate, over_rate = self.eval(test_sample,0.001)
                # print(f"[Val | {i+1}/{self.n_iter} ] EE = {(EE):.5f}, sum rate = {(sum_rate):.5f}, over rate = {(over_rate):.3f}, sigma^2=0.001")
                # val_sigma1.append(sum_rate)
                # EE, sum_rate, over_rate = self.eval(test_sample,0.01)
                # print(f"[Val | {i+1}/{self.n_iter} ] EE = {(EE):.5f}, sum rate = {(sum_rate):.5f}, over rate = {(over_rate):.3f}, sigma^2=0.01")
                # val_sigma2.append(sum_rate)
        print(f"[Best | (M,N,L,K) = ({self.M},{self.N},{self.L},{self.testing_K}) ] EE = {np.amax(val_EE):.5f}, sum rate = {np.amax(val_sum_rate):.5f}")
        # print(f"[Best | (M,N,L,K) = ({self.M},{self.N},{self.L},{self.testing_K}) ] EE = {np.amax(val_sigma1):.5f}, sigma^2=0.001")
        # print(f"[Best | (M,N,L,K) = ({self.M},{self.N},{self.L},{self.testing_K}) ] EE = {np.amax(val_sigma2):.5f}, sigma^2=0.01")

        # train_total = np.array(train_total)
        # val_EE = np.array(val_EE)
        # np.save('train_total.npy',train_total)
        # np.save('val_EE.npy',val_EE)
        # plt.figure()
        # x = np.arange(len(train_total))*self.log_interval
        # plt.plot(x,train_total)
        # x = (np.arange(len(val_EE)))*self.log_eval_interval
        # plt.plot(x,val_EE)
        # plt.xlabel('iterations')
        # plt.ylabel('EE (bps/Hz/Joule)')

        # plt.show()

    def eval(self,test_sample,sigma):
        self.model.eval()
        iteration = int(test_sample/self.batch_size)
        # iteration = 1
        num_bits = 2
        EE = []
        sum_rate_array = []
        over_array = []

        for i in range(iteration):
            user_feature, H, channel_bs_user, e = self.dataloader.load_data(self.testing_K,sigma)
            user_feature = user_feature.to(self.device)
            e = e.to(self.device)
            
            W, theta, mu, e_rc = self.model(user_feature,e)
            theta = discrete_mapping(theta,num_bits)
        
            loss, sum_rate, over_rate = cal_loss(W,theta,H,channel_bs_user,self.Pmax,mu,self.Pc_testing,self.min_rate,e_rc)
            EE.append(-loss.item())
            sum_rate_array.append(sum_rate.item())
            over_array.append(over_rate.item())
        return np.mean(EE), np.mean(sum_rate_array), np.mean(over_array)


if __name__ == '__main__':
    M = 8
    N = 30
    L = 3
    K = 3
    batch_size = 32
    for w in range(10,35,5):
        trainer = Trainer(M,N,L,K,K,batch_size,w)
        trainer.train()
