import numpy as np
import torch

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
    # for l in range(L):
    #     x = radius*np.cos(2*l*np.pi/L)
    #     y = radius*np.sin(2*l*np.pi/L)
    #     locations[l,:] = np.array([x,y])
    return locations

def element_wise_mean(x):
    batch = x.shape[0]
    return torch.mean(x,dim=1)

def element_wise_max(x):
    return torch.amax(x,dim=1)

def im2re(M):
    M1 = np.concatenate((M.real,-M.imag),axis=2)
    M2 = np.concatenate((M.imag,M.real),axis=2)
    M_mat = np.concatenate((M1,M2),axis=1)
    M_mat = torch.Tensor(M_mat)

    return M_mat

def gen_LOS(num_rev,num_trans,Rician_factor,L):
    LOS_array = []
    for l in range(L):
        AoA = np.ones((num_rev,1),dtype=np.complex128)
        AoD = np.ones((num_trans,1),dtype=np.complex128)
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

def generate_channel(M,N,L,K,batch_size,LOS_bs_ris,sigma):
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

    for sample in range(batch_size):

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

    H = np.zeros((batch_size,M,N,L,K),dtype=np.complex128)


    for k in range(K):
        for l in range(L):
            channel_ris_user_tmp = channel_ris_user[:,l,k,:,:]
            combined_ris = np.matmul(channel_bs_ris[:,l,:,:].transpose(0,2,1),channel_ris_user_tmp)
            H[:,:,:,l,k] = combined_ris

    H_imperfect = np.zeros((batch_size,M,N,L,K),dtype=np.complex128)
    if sigma > 0:
        imperfect_bs_user = np.random.normal(0,sigma,size=channel_bs_user.shape) + 1j*np.random.normal(0,sigma,size=channel_bs_user.shape)
        imperfect_ris_user = np.random.normal(0,sigma,size=channel_ris_user.shape) + 1j*np.random.normal(0,sigma,size=channel_ris_user.shape)
        imperfect_bs_user = imperfect_bs_user + channel_bs_user
        imperfect_ris_user = imperfect_ris_user + channel_ris_user
        for k in range(K):
            for l in range(L):
                channel_ris_user_tmp = imperfect_ris_user[:,l,k,:,:]
                combined_ris = np.matmul(channel_bs_ris[:,l,:,:].transpose(0,2,1),channel_ris_user_tmp)
                H_imperfect[:,:,:,l,k] = combined_ris
        return H_imperfect, imperfect_bs_user, H, channel_bs_user
    return H, channel_bs_user, H, channel_bs_user


def discrete_mapping(theta,num_bits):
    level = 2**num_bits
    phase_re =  torch.real(torch.exp(1j*torch.arange(level)/level*2*np.pi))
    phase_im =  torch.imag(torch.exp(1j*torch.arange(level)/level*2*np.pi))
    phase = torch.cat((phase_re.unsqueeze(1),phase_im.unsqueeze(1)),dim=1).to('cuda')
    for i in range(theta.shape[0]):
        for l in range(theta.shape[1]):
            for k in range(theta.shape[2]):
                temp_theta = theta[i,l,k,:]
                temp_theta = temp_theta - phase
                temp_theta = torch.norm(temp_theta,dim=1)
                index = torch.argmin(temp_theta)
                theta[i,l,k,:] = phase[index]

    return theta

def cal_loss(W, Theta, H, channel_bs_user,Pmax,mu,Pc,min_rate):
    batch_size = W.shape[0]
    M = W.shape[2]//2
    W_re = W[:,:,:M]
    W_im = W[:,:,M:]

    K = W.shape[1] - 1
    L = Theta.shape[1]
    sigma = (2e-5)**2

    gamma = []

    concat_H = torch.zeros((batch_size,K,2*M,1)).to('cuda')
    for k in range(K):
        h_k = torch.cat((torch.Tensor(channel_bs_user[:,k,:].real),torch.Tensor(channel_bs_user[:,k,:].imag)),dim=1).unsqueeze(2).to('cuda')
        concat_H_l = torch.zeros((batch_size,2*M,1)).to('cuda')
        for l in range(L):
            A = H[:,:,:,l,k]
            A_re1 = np.concatenate([A.real,A.imag],axis=2)
            A_re2 = np.concatenate([-A.imag,A.real],axis=2)
            A_re = np.concatenate([A_re1,A_re2],axis=1)
            A_re = torch.Tensor(A_re).to('cuda')

            phase_re = (Theta[:,l,:,0])
            phase_im = (Theta[:,l,:,1])
            phase_A = torch.cat((phase_re,phase_im),dim=1).unsqueeze(2)

            concat_H_l += torch.matmul(A_re,phase_A).to('cuda')
        
        concat_H[:,k,:,:] = concat_H_l + h_k
    
    common_rate = torch.zeros((batch_size,K)).to('cuda')
    for k1 in range(K):
        A = concat_H[:,k1,:,:]
        W_re_tmp = W_re[:,0,:].unsqueeze(2)
        W_im_tmp = W_im[:,0,:].unsqueeze(2)
        W_mat1 = torch.cat((W_re_tmp,W_im_tmp),dim=2)
        W_mat2 = torch.cat((-W_im_tmp,W_re_tmp),dim=2)
        W_mat = torch.cat((W_mat1,W_mat2),dim=1)

        z = torch.matmul(W_mat.transpose(2,1),A).squeeze()
        z = torch.square(z[:,0]) + torch.square(z[:,1])

        sum_power = torch.zeros(batch_size).to('cuda')
        for k2 in range(1,K+1):
            W_re_tmp = W_re[:,k2,:].unsqueeze(2)
            W_im_tmp = W_im[:,k2,:].unsqueeze(2)
            W_mat1 = torch.cat((W_re_tmp,W_im_tmp),dim=2)
            W_mat2 = torch.cat((-W_im_tmp,W_re_tmp),dim=2)
            W_mat = torch.cat((W_mat1,W_mat2),dim=1)

            z_in = torch.matmul(W_mat.transpose(2,1),A).squeeze()
            z_in = torch.square(z_in[:,0]) + torch.square(z_in[:,1])

            sum_power += z_in

        gamma = z/(sum_power+sigma)
        common_rate[:,k1] = gamma
    common_rate = torch.log2(1+torch.min(common_rate,dim=1)[0])

    private_rate = torch.zeros((batch_size,K)).to('cuda')
    for k1 in range(K):
        A = concat_H[:,k1,:,:]
        
        signal_power = []
        sum_power = torch.zeros(batch_size).to('cuda')
        for k2 in range(1,K+1):
            W_re_tmp = W_re[:,k2,:].unsqueeze(2)
            W_im_tmp = W_im[:,k2,:].unsqueeze(2)
            W_mat1 = torch.cat((W_re_tmp,W_im_tmp),dim=2)
            W_mat2 = torch.cat((-W_im_tmp,W_re_tmp),dim=2)
            W_mat = torch.cat((W_mat1,W_mat2),dim=1)

            z_in = torch.matmul(W_mat.transpose(2,1),A).squeeze()
            z_in = torch.square(z_in[:,0]) + torch.square(z_in[:,1])
            signal_power.append(z_in)
            sum_power += z_in

        gamma = signal_power[k1]/(sum_power-signal_power[k1]+sigma)
        private_rate[:,k1] = gamma
    private_rate = (torch.log2(1+private_rate))
    threshold = torch.ones(private_rate.shape).to('cuda')*min_rate
    zero = torch.zeros(private_rate.shape).to('cuda')
    threshold = torch.sum(torch.minimum(zero,private_rate-threshold),dim=1)
    over_rate = torch.sum(threshold==0)/threshold.shape[0]
    
    private_rate = torch.sum(private_rate,dim=1)
    sum_rate = common_rate + private_rate
    beta = 1
    power = mu[:,0]*Pmax + Pc
    EE = sum_rate / power
    zero = torch.zeros(sum_rate.shape).to('cuda')
    loss = -torch.mean(EE) - beta*(torch.mean(torch.minimum(zero,threshold+common_rate)))

    return loss, torch.mean(EE), torch.mean(sum_rate)

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
    



            

