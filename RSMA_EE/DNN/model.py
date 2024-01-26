import torch
import torch.nn as nn
import torch.nn.functional as F
from data import *
from utils import *

class PhaseNet(nn.Module):
    def __init__(self,M,N,L,K):
        super(PhaseNet,self).__init__()
        self.M = M
        self.N = N
        self.K = K
        self.L = L
        self.PhaseNet = nn.Sequential(
            nn.Conv2d(2*self.K,64,1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*self.M*(self.N+1),4*self.N),
            # nn.BatchNorm1d(4*self.N),
            nn.ReLU(),
            nn.Linear(4*self.N,2*self.N)
        )
    def forward(self,x):
        return self.PhaseNet(x)

class DNN(nn.Module):
    def __init__(self,M,N,L,K,Pt):
        super(DNN,self).__init__()
        self.M = M
        self.N = N
        self.K = K
        self.L = L
        self.Pt = Pt
        self.ch = 128
        
        self.PhaseNet = PhaseNet(M,N,L,K)
        # self.PhaseNet_L = []
        # for l in range(self.L):
        #     self.PhaseNet_L.append(PhaseNet(M,N,L,K))

        # self.PhaseNet_L = nn.ModuleList(self.PhaseNet_L)

        self.BeamNet = nn.Sequential(
            # nn.Conv1d(2*self.M,16,1),
            nn.Linear(2*self.K*self.M,self.ch),
            # nn.BatchNorm1d(self.ch),
            nn.ReLU(),
            # nn.Conv1d(16,16,1),
            # nn.ReLU(),
            # nn.Flatten(),
            nn.Linear(self.ch,self.ch),
            # nn.BatchNorm1d(self.ch),
            nn.ReLU(),
            nn.Linear(self.ch,2*self.M*(self.K+1))
        )

        self.powerNet = nn.Sequential(
            nn.Linear(2*self.K*self.M,self.ch),
            nn.ReLU(),
            nn.Linear(self.ch,2)
            # nn.Conv1d(2*self.M,4,1),
            # nn.ReLU(),
            # # nn.Conv1d(32,4,1),
            # # nn.ReLU(),
            # nn.Flatten(),
            # nn.Linear(4*self.K,8),
            # nn.ReLU(),
            # nn.Linear(8,2)

        )
    
    def forward(self,x,H,channel_bs_user):
        batch_size = x.shape[0]
        # print(x.shape)
        Theta = torch.zeros((batch_size,self.L,self.N,2)).to('cuda')
        # r = torch.arange(0,batch_size*self.K*self.N).reshape((batch_size,self.K,self.N)).to('cuda')
        # phase_re = (r/11).unsqueeze(3)
        # phase_im = (r/13).unsqueeze(3)
        # Theta = torch.cat((phase_re,phase_im),dim=3)
        for l in range(self.L):
            
            Rl = self.PhaseNet(x[:,l,:,:,:])
            phase_re = Rl[:,:self.N].unsqueeze(2)
            phase_im = Rl[:,self.N:].unsqueeze(2)
            A = torch.cat((phase_re,phase_im),dim=2)
            Theta[:,l,:,:] = F.normalize(A,dim=2)
        
        concat_H = torch.zeros((batch_size,self.K,2*self.M)).to('cuda')
        for k in range(self.K):
            h_k = torch.cat((torch.Tensor(channel_bs_user[:,k,:].real),torch.Tensor(channel_bs_user[:,k,:].imag)),dim=1).to('cuda')
            concat_H_l = torch.zeros((batch_size,2*self.M)).to('cuda')
            for l in range(self.L):
                A = H[:,:,:,l,k]
                A_re1 = np.concatenate([A.real,A.imag],axis=2)
                A_re2 = np.concatenate([-A.imag,A.real],axis=2)
                A_re = np.concatenate([A_re1,A_re2],axis=1)
                A_re = torch.Tensor(A_re).to('cuda')

                phase_re = (Theta[:,l,:,0])
                phase_im = (Theta[:,l,:,1])
                phase_A = torch.cat((phase_re,phase_im),dim=1).unsqueeze(2)

                concat_H_l += torch.matmul(A_re,phase_A).squeeze().to('cuda')
            concat_H[:,k,:] = concat_H_l + h_k
        
        concat_H = concat_H.reshape((batch_size,-1))
        # concat_H = concat_H.transpose(2,1)
        
        W = self.BeamNet(concat_H)
        W = W.reshape((batch_size,self.K+1,2*self.M))
        mu = self.powerNet(concat_H)
        mu = F.softmax(mu,1)
        # mu = torch.ones((batch_size,1)).to('cuda')
        # W = F.normalize(W,dim=1)*torch.sqrt(mu[:,0]).reshape(-1,1,1)*np.sqrt(self.Pt/(self.K+1))
        # W = W.transpose(2,1)
        W = W.reshape((batch_size,-1))
        W = F.normalize(W,dim=1)*np.sqrt(self.Pt)*torch.sqrt(mu[:,0]).reshape(-1,1)
        W = W.reshape((batch_size,self.K+1,-1))

        return W, Theta, mu
        

if __name__ == '__main__':
    M = 8
    N = 50
    L = 8
    K = 4
    batch_size = 32
    dataloader = MyDataLoader(M,N,L,batch_size)
    user_feature, H, channel_bs_user, e = dataloader.load_data(K)
    model = initial_layer(M,N,L,K,32)
    uk, uc, rl = model(user_feature,e)

    update_layer = node_update_layer(32,M,N,L,K,32)
    uk, uc, rl = update_layer(uk, uc, rl, e)

    uk = uk.to('cuda')
    uc = uc.to('cuda')
    rl = rl.to('cuda')
    cal_adj = cal_EE(M,N,L,K,64).to('cuda')
    adj = cal_adj(uk,uc,rl,e)
