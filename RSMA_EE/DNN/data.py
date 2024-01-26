from utils import *
import time
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

class MyDataLoader(Dataset):
    def __init__(self,M,N,L,batch_size):
        super(MyDataLoader,self).__init__()
        self.M = M
        self.N = N
        self.L = L
        self.batch_size = batch_size
        self.LOS_bs_ris = gen_LOS(self.N,self.M,10,self.L)
    
    def load_data(self,K,sigma):
        self.K = K
        H, channel_bs_user, perfect_H, perfect_bs_user = generate_channel(self.M,self.N,self.L,self.K,self.batch_size,self.LOS_bs_ris,sigma)
        batch_size = H.shape[0]
        user_feature = np.zeros((H.shape[0],self.L,2*self.K,self.M,self.N+1))
        e = np.zeros((H.shape[0],self.L,self.K))
        for l in range(self.L):
            temp_H = H[:,:,:,l,:].transpose(0,3,1,2)
            user_feature[:,l,:self.K,:,:self.N] = temp_H.real
            user_feature[:,l,self.K:,:,:self.N] = temp_H.imag

            temp_bs_user = channel_bs_user
            user_feature[:,l,:self.K,:,self.N] = temp_bs_user.real
            user_feature[:,l,self.K:,:,self.N] = temp_bs_user.imag

        user_feature = torch.Tensor(user_feature)
        user_feature = F.normalize(user_feature,dim=2)


        return user_feature, perfect_H, perfect_bs_user

if __name__ == '__main__':
    M = 8
    N = 50
    L = 4
    K = 4
    batch_size = 32
    dataloader = MyDataLoader(M,N,L,batch_size)
    user_feature, H, channel_bs_user = dataloader.load_data(K)

        

