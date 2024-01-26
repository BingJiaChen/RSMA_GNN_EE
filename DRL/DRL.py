import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import optuna
import time
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)



def cal_loss(W, Theta, H, channel_bs_user,Pmax,mu,Pc,min_rate,e_rc,M,N,L,K):#array dimension require modified 
    W_re = W[:(K+1)*M]
    W_re = (W_re.reshape((1,K+1,M)))
    W_im = W[(K+1)*M:]
    W_im = (W_im.reshape((1,K+1,M)))
    p = torch.pow(W_re,2) + torch.pow(W_im,2)
    channel_bs_user = channel_bs_user.reshape((1,K,-1))
    H = H.reshape((1,M,N,L,K))
    Theta = Theta.reshape((1,L,N))
    e_rc = torch.FloatTensor(e_rc).to('cuda')
    mu = (mu.reshape((1,-1)))
    sigma = (2e-5)**2

    gamma = []

    concat_H = torch.zeros((1,K,2*M,1)).to('cuda')
    for k in range(K):
        h_k = torch.cat((torch.Tensor(channel_bs_user[:,k,:].real),torch.Tensor(channel_bs_user[:,k,:].imag)),dim=1).unsqueeze(2).to('cuda')
        concat_H_l = torch.zeros((1,2*M,1)).to('cuda')
        for l in range(L):
            A = H[:,:,:,l,k]
            A_re1 = np.concatenate([A.real,A.imag],axis=2)
            A_re2 = np.concatenate([-A.imag,A.real],axis=2)
            A_re = np.concatenate([A_re1,A_re2],axis=1)
            A_re = torch.Tensor(A_re).to('cuda')

            phase_re = (Theta[:,l,:].real)
            phase_im = (Theta[:,l,:].imag)
            phase_A = torch.cat((phase_re,phase_im),dim=1).unsqueeze(2)

            concat_H_l += torch.matmul(A_re,phase_A).to('cuda')

        concat_H[:,k,:,:] = concat_H_l + h_k

    common_rate = torch.zeros((1,K)).to('cuda')
    for k1 in range(K):
        A = concat_H[:,k1,:,:]
        W_re_tmp = W_re[:,0,:].unsqueeze(2)
        W_im_tmp = W_im[:,0,:].unsqueeze(2)
        W_mat1 = torch.cat((W_re_tmp,W_im_tmp),dim=2)
        W_mat2 = torch.cat((-W_im_tmp,W_re_tmp),dim=2)
        W_mat = torch.cat((W_mat1,W_mat2),dim=1)

        z = torch.matmul(W_mat.transpose(2,1),A).squeeze(2)
        z = torch.square(z[:,0]) + torch.square(z[:,1])

        sum_power = torch.zeros(1).to('cuda')
        for k2 in range(1,K+1):
            W_re_tmp = W_re[:,k2,:].unsqueeze(2)
            W_im_tmp = W_im[:,k2,:].unsqueeze(2)
            W_mat1 = torch.cat((W_re_tmp,W_im_tmp),dim=2)
            W_mat2 = torch.cat((-W_im_tmp,W_re_tmp),dim=2)
            W_mat = torch.cat((W_mat1,W_mat2),dim=1)

            z_in = torch.matmul(W_mat.transpose(2,1),A).squeeze(2)
            z_in = torch.square(z_in[:,0]) + torch.square(z_in[:,1])

            sum_power += z_in

        gamma = z/(sum_power+sigma)
        common_rate[:,k1] = gamma
    common_rate = torch.log2(1+torch.min(common_rate))

    private_rate = torch.zeros((1,K)).to('cuda')
    for k1 in range(K):
        A = concat_H[:,k1,:,:]

        signal_power = []
        sum_power = torch.zeros(1).to('cuda')
        for k2 in range(1,K+1):
            W_re_tmp = W_re[:,k2,:].unsqueeze(2)
            W_im_tmp = W_im[:,k2,:].unsqueeze(2)
            W_mat1 = torch.cat((W_re_tmp,W_im_tmp),dim=2)
            W_mat2 = torch.cat((-W_im_tmp,W_re_tmp),dim=2)
            W_mat = torch.cat((W_mat1,W_mat2),dim=1)

            z_in = torch.matmul(W_mat.transpose(2,1),A).squeeze(2)
            z_in = torch.square(z_in[:,0]) + torch.square(z_in[:,1])
            signal_power.append(z_in)
            sum_power += z_in

        gamma = signal_power[k1]/(sum_power-signal_power[k1]+sigma)
        private_rate[:,k1] = gamma

    private_rate = (torch.log2(1+private_rate))
    # common_rate = common_rate.unsqueeze(1).repeat(1,K) * e_rc
    # sum_rate = common_rate + private_rate
    # threshold = (torch.ones(sum_rate.shape)*min_rate).to('cuda')
    # zero = torch.zeros(sum_rate.shape).to('cuda')
    # threshold = torch.sum(torch.minimum(zero,sum_rate-threshold),dim=1)

    beta = 1
    sum_rate = torch.sum(private_rate) + common_rate
    power = mu[:,0]*Pmax + Pc
    EE = sum_rate / power
    loss = torch.mean(EE)
    # over_rate = torch.sum(threshold==0)/threshold.shape[0]

    return loss, torch.mean(sum_rate), torch.mean(sum_rate)

class Channel():
    def __init__(self,num_trans,num_rev,factor):
        self.num_trans = num_trans
        self.num_rev = num_rev
        self.factor = factor
        self.AoA = np.ones((num_rev,1),dtype=np.complex128)
        self.AoD = np.ones((num_trans,1),dtype=np.complex128)
        self.mat = np.zeros((num_rev,num_trans),dtype=np.complex128)
        self.NLOS = np.zeros((num_rev,num_trans),dtype=np.complex128)
        self.LOS = np.zeros((num_rev,num_trans),dtype=np.complex128)

    def generate_value(self,mean_NLOS,cov_NLOS,LOS):
        self.LOS = LOS
        R_NLOS = np.linalg.cholesky(cov_NLOS)
        mat_real_NLOS = np.ones((self.num_rev,self.num_trans))*mean_NLOS + np.matmul(np.random.randn(self.num_rev,self.num_trans),R_NLOS)
        mat_imag_NLOS = np.ones((self.num_rev,self.num_trans))*mean_NLOS + np.matmul(np.random.randn(self.num_rev,self.num_trans),R_NLOS)
        self.NLOS = np.sqrt(1/(self.factor+1))*(mat_real_NLOS+1j*mat_imag_NLOS)/np.sqrt(2)

        self.mat = self.NLOS + self.LOS

    def large_scale_loss(self,fading_NLOS,exp_NLOS,fading_LOS,exp_LOS,dist):
        self.large_scale_fading_NLOS = fading_NLOS*dist**(-exp_NLOS)
        self.large_scale_fading_LOS = fading_LOS*dist**(-exp_LOS)
        self.ori_mat = self.large_scale_fading_NLOS*self.NLOS+self.large_scale_fading_LOS*self.LOS
        self.mat = self.large_scale_fading_NLOS*self.NLOS+self.large_scale_fading_LOS*self.LOS

        return self.mat

def gen_location_user(K,l):
    center = np.array([200,0])
    locations = np.zeros((K,2))
    for k in range(K):
        x = l*np.cos(np.random.uniform(0,np.pi))
        y = l*np.sin(np.random.uniform(0,np.pi))
        locations[k,:] = center + np.array([x,y])
    return locations

def gen_location_RIS(L):
    x = 100
    locations = np.zeros((L,2))
    y = np.array([0,-10,10,-20,20])
    locations[:,0] = x
    locations[:,1] = y[:L]
    return locations

def gen_LOS(num_rev,num_trans,Rician_factor,L):
    LOS_array = []
    for l in range(L):
        AoA = np.ones((num_rev,1),dtype=np.complex128)
        AoD = np.ones((num_trans,1),dtype=np.complex128)
        # angle_AoA = 2*np.pi*np.random.uniform(0,1)
        angle_AoA = 2*np.pi*(l+1)/11
        for n in range(1,num_rev):
            AoA[n,:] = np.exp(1j*n*np.pi*np.sin(angle_AoA))
        angle_AoD = 2*np.pi*(l+1)/11
        for n in range(1,num_trans):
            AoD[n,:] = np.exp(1j*n*np.pi*np.sin(angle_AoD))
        mat_LOS = np.dot(AoA,np.conj(AoD.T))
        LOS = np.sqrt(Rician_factor/(Rician_factor+1))*mat_LOS
        if L == 1:
            return LOS
        LOS_array.append(LOS)

    return LOS_array

def generate_channel(M,N,L,K,RSMA_batch_size,LOS_bs_ris,sigma):
    loc_RIS = gen_location_RIS(L)
    loc_BS = np.array([0,0])
    loc_user = gen_location_user(K,20)
    Rician_factor = 10
    background_noise = 2e-12
    fading_BS_users = 10**(-4.5)
    path_loss_exp_BS_user = 3.5
    fading_BS_RIS = 10**(-0.5)
    fading_RIS_users = 10**(-0.5)
    path_loss_exp_LOS = 2
    path_loss_exp_NLOS = 2.5

    channel_bs_ris = []
    channel_ris_user = []
    channel_bs_user = []
    theta = []

    epsilon = 0.01

    for sample in range(RSMA_batch_size):

        tmp_bs_ris = []
        for l in range(L):
            H_BS_RIS = Channel(M,N,Rician_factor)
            H_BS_RIS.generate_value(0,np.eye(M),LOS_bs_ris[l])
            H_BS_RIS = H_BS_RIS.large_scale_loss(fading_BS_RIS,path_loss_exp_NLOS,fading_BS_RIS,path_loss_exp_LOS,np.linalg.norm(loc_RIS[l]))
            tmp_bs_ris.append(H_BS_RIS)
        channel_bs_ris.append(tmp_bs_ris)

        tmp_ris_user = [[] for l in range(L)]
        tmp_bs_user = []
        for k in range(K):

            for l in range(L):
                h_LOS = gen_LOS(1,N,Rician_factor,1)
                h_RIS = Channel(N,1,Rician_factor)
                h_RIS.generate_value(0,np.eye(N),h_LOS)
                h_RIS = h_RIS.large_scale_loss(fading_RIS_users,path_loss_exp_NLOS,fading_RIS_users,path_loss_exp_LOS,np.linalg.norm(loc_user[k,:]-loc_RIS[l,:]))
                tmp_ris_user[l].append(np.diag(h_RIS[0]))

            h_LOS = gen_LOS(1,M,0,1)
            h_bs = Channel(M,1,0)
            h_bs.generate_value(0, np.eye(M), h_LOS)
            h_bs = h_bs.large_scale_loss(fading_BS_users,path_loss_exp_BS_user,0,0,np.linalg.norm(loc_user[k,:]))
            tmp_bs_user.append(h_bs[0])

        channel_ris_user.append(tmp_ris_user)
        channel_bs_user.append(tmp_bs_user)

    scale = -7
    background_noise = background_noise/10**(scale)

    channel_bs_ris = np.array(channel_bs_ris)/np.sqrt(10**scale)
    channel_ris_user = np.array(channel_ris_user)/np.sqrt(10**scale)
    channel_bs_user = np.array(channel_bs_user)/10**scale

    H = np.zeros((RSMA_batch_size,M,N,L,K),dtype=np.complex128)


    for k in range(K):
        for l in range(L):
            channel_ris_user_tmp = channel_ris_user[:,l,k,:,:]
            combined_ris = np.matmul(channel_bs_ris[:,l,:,:].transpose(0,2,1),channel_ris_user_tmp)
            H[:,:,:,l,k] = combined_ris

    return H, channel_bs_user, H, channel_bs_user

def discrete_mapping(theta,M,N,L,K):

    Distheta = np.zeros(L*N, dtype=complex)
    for m in range(theta.shape[0]):
        phase = theta[m]
        if phase >= 0.875 and phase < 0.125:
            Distheta[m] = 1
        elif phase >= 0.125 and phase < 0.375:
            Distheta[m] = 1j
        elif phase >= 0.375 and phase < 0.625:
            Distheta[m] =-1
        else:
            Distheta[m] = -1j
    return Distheta

def gen_Laplacian(adj):
    ns = adj.shape[1]
    bs = adj.shape[0]
    iden = torch.eye(ns)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    iden = iden.unsqueeze(0).repeat(bs,1,1).to(device)
    adj[iden.bool()] = 0
    D = torch.diag_embed(torch.pow(adj.sum(axis=2),-0.5))
    sym_norm_lap = iden - torch.bmm(D,torch.bmm(adj,D))

    return sym_norm_lap

def gentrainchannel(M, N, L, K, RSMA_batch_size, LOS_bs_ris):

    H, D,_, _ = generate_channel(M, N, L, K, RSMA_batch_size, LOS_bs_ris, 0)
    Hreal = H[0].flatten().real
    Himg = H[0].flatten().imag
    Dreal = D[0].flatten().real
    Dimg = D[0].flatten().imag

    tmpreal = np.concatenate((Hreal, Himg))
    tmpimg = np.concatenate((Dreal, Dimg))
    final = np.concatenate((tmpreal, tmpimg))
    return final

def genrewardchannel(oneDimChan,M,N,L,K):
    Hreal  = oneDimChan[0:M*N*L*K]
    Himg = oneDimChan[M*N*L*K:M*N*L*K*2]
    Dreal = oneDimChan[M*N*L*K*2:M*N*L*K*2+K*M]
    Dimg = oneDimChan[M*N*L*K*2+K*M:]
    Hfinal = Hreal + 1j*Himg
    Dfinal = Dreal + 1j*Dimg

    return Hfinal, Dfinal






# RSMA Environment
class RSMAEnv():

    #action space = phaseshift L*N + beamforming (K+1) * (M) + power(1) + rate allocation(K)
    def __init__(self, state_dim, action_dim, max_action, rho,M,N,L,K, steps):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.LOS_bs_ris = gen_LOS(N,M,10,L)
        self.statestart = gentrainchannel(M, N, L, K, 1, self.LOS_bs_ris)
        self.state = self.statestart
        self.storeState = np.zeros(steps, dtype = np.complex64)
        self.rho = rho
        self.time_step = 0
        self.Pc_training = 1/0.8 + 0.001*N*L
        self.Pc_testing = 1/0.8 + 0.001*N*L
        self.Pt = 40
        self.Pmax = 10**((self.Pt-30)/10)
        self.min_rate = 5
        self.device = 'cuda'
        self.M = M
        self.N = N
        self.L = L
        self.K = K
        self.steps = steps
        #self.dataloader = MyDataLoader(M, N, L, RSMA_batch_size)
        


    def reset(self):
        self.state = self.statestart
        self.time_step = 0
        return self.state

    def step(self, action):
        self.time_step += 1
        reward, sumrate = self.getReward(action, self.state)
        self.state = self.ARNextChannel(self.state)
        done = self.time_step >= self.steps
        return self.state, reward, done, sumrate

    def getReward(self, action, state):
        H, channel_bs_user = genrewardchannel(state,self.M,self.N,self.L,self.K)
        theta_re = torch.FloatTensor(action[:self.L*self.N]).unsqueeze(1).to('cuda')
        theta_im = torch.FloatTensor(action[self.L*self.N:self.L*self.N*2]).unsqueeze(1).to('cuda')
        # theta_re = torch.arange(self.L*self.N).unsqueeze(1).to('cuda')/11
        # theta_im = torch.arange(self.L*self.N).unsqueeze(1).to('cuda')/13
        theta = torch.cat((theta_re,theta_im),dim=1)
        theta = F.normalize(theta,dim=1)
        theta = theta[:,0] + 1j*theta[:,1]

        W = action[self.L*self.N*2:self.L*self.N*2+(self.K+1)*self.M*2]
        # W = action[:(self.K+1)*self.M*2]
        mu = action[self.L*self.N*2+(self.K+1)*self.M*2:self.L*self.N*2+(self.K+1)*self.M*2 + 2]
        # mu = action[(self.K+1)*self.M*2:(self.K+1)*self.M*2 + 2]
        mu = torch.FloatTensor(mu).to('cuda')
        mu = F.softmax(mu,dim=0)
        W = torch.FloatTensor(W).to('cuda')
        W = F.normalize(W,dim=0)*torch.sqrt(self.Pmax*mu[0])
        # W = F.normalize(W,dim=0)*np.sqrt(self.Pmax)
        e_rc = action[self.L*self.N*2+(self.K+1)*self.M*2 + 2:]
        # e_rc = np.ones(self.K)
        e_rc = (np.abs(e_rc) / np.sum(np.abs(e_rc))) * 1.0

        loss, sum_rate, over_rate = cal_loss(W,theta,H,channel_bs_user,self.Pmax,mu,self.Pc_testing,self.min_rate,e_rc,self.M,self.N,self.L,self.K)
        EE = loss
        sumRate = sum_rate.item()
        overRate = over_rate.item()
        return EE, sumRate

    def ARNextChannel(self, state):
        state  = self.rho * state + np.sqrt(1-self.rho**2)*gentrainchannel(self.M, self.N, self.L, self.K, 1, self.LOS_bs_ris)
        return state





# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        # self.C = nn.Sequential(
        #     nn.Linear(state_dim, hidden_dim*4),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim*4, hidden_dim*2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim*2, action_dim)
        # )
        self.fc1 = nn.Linear(state_dim, hidden_dim*4)
        self.bn1 = nn.BatchNorm1d(hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, action_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, state, action):
        state = F.normalize(state,dim=1)
        state = torch.cat((state,action),dim=1)
        # action = self.C(state)
        x = torch.relu(self.bn1(self.fc1(state)))
        x = torch.relu(self.bn2(self.fc2(x)))
        action = torch.tanh(self.fc3(x))
        # action = (self.fc3(x))
        return action

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim*4)
        self.bn1 = nn.BatchNorm1d(hidden_dim*4)
        self.fc2 = nn.Linear(action_dim + hidden_dim*4, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, state, action):
        state = F.normalize(state,dim=1)
        state = torch.cat((state,action),dim=1)
        s1 = torch.relu(self.bn1(self.fc1(state)))
        x = torch.cat([s1, action], dim=1)
        x = torch.relu(self.bn2(self.fc2(x)))
        Q_value = self.fc3(x)
        return Q_value

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, pre_action, state, action, reward, next_state, done):
        self.memory.append((pre_action, state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        pre_action, state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(pre_action),
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.tensor(reward, dtype=torch.float32),
            torch.FloatTensor(next_state),
            torch.tensor(done, dtype=torch.float32)
        )

    def reset(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, actor_lr, critic_lr, gamma, tau, hidden_dim):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.target_actor = Actor(state_dim, action_dim, hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(1000)
        self.batch_size = 64

    def select_action(self, state, action, noise=0):
        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        action = self.actor(state,action).detach().cpu().numpy()[0]
        action += noise
        return action

    def train(self):
        self.actor.train()
        if len(self.replay_buffer) < self.batch_size:
            return

        pre_action, states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # Compute target Q-values
        with torch.no_grad():
            target_actions = self.target_actor(next_states, actions)
            target_Q = self.target_critic(next_states, target_actions)
            target_Q = rewards.reshape((-1,1)) + self.gamma * torch.matmul((1.0 - dones),target_Q)
            # target_Q = rewards.reshape((-1,1)) + self.gamma * target_Q


        # Update critic
        Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(states, self.actor(states, pre_action)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update the target networks
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))








class ddpg_optimize:
    def __init__(self,state_dim,action_dim,max_action,rho,steps,M,N,L,K):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.steps = steps
        self.rho = rho
        self.M = M
        self.N = N
        self.L = L
        self.K = K
        self.env = RSMAEnv(self.state_dim, self.action_dim, self.max_action, self.rho,M,N,L,K,self.steps)

    def train_ddpg_with_optuna(self,trial):
        state_dim = self.env.state_dim
        action_dim = self.env.action_dim
        max_action = self.env.max_action

        actor_lr = trial.suggest_loguniform('actor_lr', 1e-5, 1e-2)
        critic_lr = trial.suggest_loguniform('critic_lr', 1e-5, 1e-2)
        tau = trial.suggest_loguniform('tau', 1e-3, 1e-1)
        gamma = trial.suggest_uniform('gamma', 0.9, 0.9999)
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])

        agent = DDPGAgent(self.state_dim, self.action_dim, self.max_action, actor_lr, critic_lr, gamma, tau, hidden_dim)
        mu = 0
        noiseSigma = 1
        num_episodes = 100
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            sum_rate = 0
            num = 0

            while True:
                noise = np.random.normal(mu,noiseSigma,size=self.action_dim)
                action = agent.select_action(state,noise)
                next_state, reward, done, sumrate = self.env.step(action)
                agent.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                num += 1
                agent.train()

                if done:
                    episode_reward = reward
                    sum_rate = sumrate
                    break

            print(f"Episode: {episode + 1}, Reward: {episode_reward}, Sum rate: {sum_rate}")

        # Return the cumulative reward or other performance metric
        return episode_reward

    def optimize_hyperparameters(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.train_ddpg_with_optuna, n_trials=2)
        return study.best_params


if __name__ == "__main__":

    RANDOMSEED = 1
    np.random.seed(RANDOMSEED)
    M = 4
    N = 5
    L = 1
    K = 3
    rho = 0.99995
    # rho = 1
    Pt = 40
    Pmax = 10**((Pt-30)/10)
    num_bits = 2
    RSMA_batch_size = 1
    action_dim = 2*L*N + M * (K + 1) * 2 + 2 + K
    state_dim = 2*(M*N*L*K + M*K) + action_dim
    max_action = 1
    steps = 20
    mu = 0
    noiseSigma = 1

    # autotuner = ddpg_optimize(state_dim,action_dim,max_action,rho,steps,M,N,L,K)
    # best_params = autotuner.optimize_hyperparameters()
    # print("Best parameters:", best_params)


    # agent = DDPGAgent(state_dim, action_dim, max_action, best_params['actor_lr'], best_params['critic_lr'], best_params['gamma'], best_params['tau'], best_params['hidden_dim'])
    # agent = DDPGAgent(state_dim, action_dim, max_action, 1e-4,1e-3,0.99,0.001,256)


    num_episodes = 2000
    # reward_array = []
    # env = RSMAEnv(state_dim, action_dim, max_action, rho,M,N,L,K,steps)
    # for episode in range(num_episodes):
    #     state = env.reset()
    #     pre_action = np.random.normal(mu,noiseSigma,size=action_dim) 
    #     # agent.replay_buffer.reset()
    #     episode_reward = 0
    #     sum_rate = 0
    #     num = 0
    #     if (episode+1)%100 == 0:
    #         noiseSigma*=0.995

    #     while True:
    #         noise = np.random.normal(mu,noiseSigma,size=action_dim)
    #         # noise = np.zeros(action_dim)
    #         action = agent.select_action(state, pre_action, noise)
    #         next_state, reward, done, sumrate = env.step(action)
    #         agent.replay_buffer.push(pre_action, state, action, reward, next_state, done)
    #         state = next_state
    #         pre_action = action
    #         episode_reward += reward.item()
            
    #         sum_rate += sumrate
    #         num += 1

    #         agent.train()

    #         if done:
    #             break
    #     reward_array.append(episode_reward/num)

    #     print(f"Episode: {episode + 1}, Reward: {episode_reward/num}, Sum rate: {sum_rate/num}")
    # reward_array = np.array(reward_array)
    x = np.arange(num_episodes)
    # np.save('reward.npy',reward_array)
    reward_array = np.load('reward.npy')

    plt.figure()
    plt.plot(x,reward_array)
    plt.xlabel('Episodes')
    plt.ylabel('EE (bps/Hz/Joule)')
    plt.ylim([0,2])
    plt.show()