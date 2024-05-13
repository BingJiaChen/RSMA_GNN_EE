import torch
import torch.nn as nn
import torch.nn.functional as F
from data import *
from utils import *

class initial_layer(nn.Module):
    def __init__(self,M,N,L,ch):
        super(initial_layer,self).__init__()
        self.ch = ch
        self.M = M
        self.N = N
        self.L = L
        self.in_dim = 2*M*(N+1)

        self.fu = nn.Sequential(
            nn.Linear(self.in_dim,self.ch*2),
            nn.ReLU(),
            nn.Linear(self.ch*2,self.ch)
        )

        self.f = nn.Sequential(
            nn.Linear(self.ch,self.ch),
            nn.ReLU(),
            nn.Linear(self.ch,self.ch)
        )

    def forward(self,H,e):
        batch_size = H.shape[0]
        uk = self.fu(H)
        uk = torch.mean(uk,dim=1)

        e_h = F.normalize(e,p=1,dim=2)
        rl = torch.matmul(e_h,uk)
        rl = self.f(rl)

        uc = torch.mean(uk,dim=1)

        return uk, uc, rl

class node_update_layer(nn.Module):
    def __init__(self,in_dim,M,N,L,ch):
        super(node_update_layer,self).__init__()
        self.M = M
        self.N = N
        self.L = L
        self.ch = ch
        self.in_dim_p = in_dim*3
        self.in_dim_c = in_dim*2
        self.in_dim_l = in_dim*2

        self.fu = nn.Sequential(
            nn.Linear(self.in_dim_p,self.ch*2),
            nn.ReLU(),
            nn.Linear(self.ch*2,self.ch)
        )

        self.f = nn.Sequential(
            nn.Linear(self.in_dim_l,self.ch*2),
            nn.ReLU(),
            nn.Linear(self.ch*2,self.ch)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.in_dim_c,self.ch*2),
            nn.ReLU(),
            nn.Linear(self.ch*2,self.ch)
        )

        self.edge_update = nn.Sequential(
            nn.Linear(in_dim+self.ch,self.ch),
            nn.ReLU(),
            nn.Linear(self.ch,1)
        )

    def forward(self,uk, uc, rl, e, e_rc):
        batch_size = uk.shape[0]
        self.K = uk.shape[1]
        uk_update = torch.zeros((batch_size,self.K,self.ch+uk.shape[-1])).to('cuda')

        e_l = e.transpose(2,1)
        e_l = F.normalize(e_l,p=1,dim=2)
        mean_uk = torch.matmul(e_l,rl)
        for k in range(self.K):
            if k != self.K:
                max_user = torch.amax(torch.cat((uk[:,:k,:],uk[:,k+1:,:]),dim=1),dim=1)
            else:
                max_user = torch.amax(uk[:,:k,:],dim=1)
            
            tmp_uk = self.fu(torch.cat((uk[:,k,:],max_user,mean_uk[:,k,:]),dim=1))
            uk_update[:,k,:] = torch.cat((tmp_uk,uk[:,k,:]),dim=1)
        
        e_h = F.normalize(e,p=1,dim=2)
        mean_rl = torch.matmul(e_h,uk)
        tmp_rl = self.f(torch.cat((rl,mean_rl),dim=2))
        rl_update = torch.cat((tmp_rl,rl),dim=2)

        mean_uc = torch.matmul(uk.transpose(2,1),e_rc.unsqueeze(2)).squeeze()
        tmp_uc = self.fc(torch.cat((mean_uc,uc),dim=1))
        uc_update = torch.cat((tmp_uc,uc),dim=1)
        e_rc_update = self.edge_update(torch.abs(uc_update.unsqueeze(1).repeat(1,self.K,1)-uk_update))
        e_rc_update = e_rc_update.squeeze()
        e_rc_update = F.softmax(e_rc_update,dim=1)

        return uk_update, uc_update, rl_update, e_rc_update

class cal_EE(nn.Module):
    def __init__(self,M,N,L,in_dim):
        super(cal_EE,self).__init__()
        self.M = M
        self.N = N
        self.L = L
        self.binary = nn.Sequential(
            nn.Linear(in_dim,in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2,2)
        )

    def forward(self,uk,uc,rl,e,e_rc):
        batch_size = uk.shape[0]
        self.K = uk.shape[1]
        adj = torch.zeros((batch_size,self.L+self.K+1,self.L+self.K+1)).to('cuda')
        iden = torch.eye(self.L)
        iden = iden.unsqueeze(0).repeat(batch_size,1,1).to('cuda')
        # adj[:,:self.L,:self.L] = iden
        # adj[:,:self.L,self.L:self.L+self.K] = F.normalize(e,p=1,dim=2)
        # adj[:,self.L:self.L+self.K,:self.L] = F.normalize(e,p=1,dim=2).transpose(2,1)
        adj[:,:self.L,self.L:self.L+self.K] = e
        adj[:,self.L:self.L+self.K,:self.L] = e.transpose(2,1)
        iden = torch.eye(self.K)
        iden = iden.unsqueeze(0).repeat(batch_size,1,1).to('cuda')
        u2u = torch.ones((batch_size,self.K,self.K))
        u2u[iden.bool()] = 1
        adj[:,self.L:self.L+self.K,self.L:self.L+self.K] = u2u
        # adj[:,self.L:self.L+self.K,self.L+self.K] = e_rc
        adj[:,self.L+self.K,self.L:self.L+self.K] = e_rc
        adj[:,self.L+self.K,self.L+self.K] = 1

        laplacian = gen_Laplacian(adj)
        laplacian = torch.transpose(laplacian,2,1)
        X = torch.cat((rl,uk,uc.unsqueeze(1)),dim=1)
        til_L = torch.matmul(laplacian,X)
        output = self.binary(torch.mean(til_L[:,:,:],dim=1))
        output = F.softmax(output,1)

        return output


class readout(nn.Module):
    def __init__(self,M,N,L,Pt,in_dim):
        super(readout,self).__init__()
        self.M = M
        self.N = N
        self.L = L
        self.Pt = Pt
        self.in_dim = in_dim

        self.fu = nn.Linear(self.in_dim,self.M*2)
        self.f = nn.Linear(self.in_dim,self.N*2)
        self.fc = nn.Linear(self.in_dim,self.M*2)

    def forward(self,uk,uc,rl,mu):
        batch_size = uk.shape[0]
        self.K = uk.shape[1]
        Wu = self.fu(uk)
        Wc = self.fc(uc)
        W = torch.cat((Wc.unsqueeze(1),Wu),dim=1)
        W = W.reshape((batch_size,-1))
        W = F.normalize(W,dim=1)*np.sqrt(self.Pt)*torch.sqrt(mu[:,0]).reshape(-1,1)
        W = W.reshape((batch_size,self.K+1,-1))

        Rl = self.f(rl)
        phase_re = Rl[:,:,:self.N].unsqueeze(3)
        phase_im = Rl[:,:,self.N:].unsqueeze(3)
        # r = torch.arange(0,batch_size*self.K*self.N).reshape((batch_size,self.K,self.N)).to('cuda')
        # phase_re = (r/11).unsqueeze(3)
        # phase_im = (r/13).unsqueeze(3)
        phase = torch.cat((phase_re,phase_im),dim=3)
        phase = F.normalize(phase,dim=3)

        return W, phase


class node_update(nn.Module):
    def __init__(self,M,N,L,D,Pmax,b,ch):
        super(node_update,self).__init__()
        self.N = N
        self.D = D
        self.M = M
        self.L = L
        self.ch = ch
        self.Pmax = Pmax
        self.b = b
        self.init_user = initial_layer(M,N,L,ch)
        update_list = []
        for d in range(D):
            update_list.append(node_update_layer(ch*(d+1),M,N,L,ch))
        self.update_list = nn.ModuleList(update_list)
        self.cal_EE = cal_EE(M,N,L,ch*(D+1))
        self.readout = readout(M,N,L,Pmax,ch*(D+1))
    
    def forward(self,user_feature,e):
        batch_size = user_feature.shape[0]
        uk, uc, rl = self.init_user(user_feature,e)
        e_rc = (torch.ones((batch_size,uk.shape[1]))/uk.shape[1]).to('cuda')
        for i, _ in enumerate(self.update_list):
            update_layer = self.update_list[i]
            uk, uc, rl, e_rc = update_layer(uk, uc, rl, e, e_rc)
            uk = uk.to('cuda')
            uc = uc.to('cuda')
            rl = rl.to('cuda')
            e_rc = e_rc.to('cuda')

        mu = self.cal_EE(uk,uc,rl,e,e_rc)
        # mu = torch.ones((batch_size,1)).to('cuda')
        W, theta = self.readout(uk,uc,rl,mu)

        return W, theta, mu, e_rc

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

    update_layer = node_update_layer(32,M,N,L,32)
    uk, uc, rl = update_layer(uk, uc, rl, e)

    uk = uk.to('cuda')
    uc = uc.to('cuda')
    rl = rl.to('cuda')
    cal_adj = cal_EE(M,N,L,K,64).to('cuda')
    adj = cal_adj(uk,uc,rl,e)
