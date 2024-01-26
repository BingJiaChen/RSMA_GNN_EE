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
        user_feature = np.zeros((H.shape[0],self.L,self.K,2*self.M,self.N+1))
        e = np.zeros((H.shape[0],self.L,self.K))
        # e = np.ones((H.shape[0],self.L,self.K))
        for k in range(self.K):
            for l in range(self.L):
                temp_H = H[:,:,:,l,k]
                user_feature[:,l,k,:self.M,:self.N] = temp_H.real
                user_feature[:,l,k,self.M:,:self.N] = temp_H.imag

                w_H = np.matmul(temp_H,np.conj(temp_H.transpose(0,2,1)))
                w_H = np.trace(w_H,axis1=1,axis2=2)
                e[:,l,k] = np.real(w_H)

                temp_bs_user = channel_bs_user[:,k,:]
                user_feature[:,l,k,:self.M,self.N] = temp_bs_user.real
                user_feature[:,l,k,self.M:,self.N] = temp_bs_user.imag

                temp_bs_user = temp_bs_user.reshape((batch_size,1,-1))
                w_h = np.matmul(temp_bs_user,np.conj(temp_bs_user.transpose(0,2,1)))
                e[:,l,k] = e[:,l,k] + np.real(w_h[:,0,0])

        user_feature = user_feature.reshape((batch_size,self.L,self.K,-1))
        user_feature = torch.Tensor(user_feature)
        user_feature = F.normalize(user_feature,dim=2)

        e = torch.Tensor(e)

        return user_feature, perfect_H, perfect_bs_user, e

if __name__ == '__main__':
    M = 8
    N = 50
    L = 4
    K = 4
    batch_size = 32
    dataloader = MyDataLoader(M,N,L,batch_size)
    user_feature, H, channel_bs_user = dataloader.load_data(K)

        

