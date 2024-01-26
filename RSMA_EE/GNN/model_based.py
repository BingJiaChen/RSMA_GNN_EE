import cvxpy as cp
import numpy as np
from utils import *
import math
import warnings
warnings.filterwarnings("ignore")

# parameters setup ----------------------------------------------

M = 8
N = 20
L = 3
K = 3

Pmax = 10
sigma = (2e-5)**2
C = 1
Rk = 1
Pc = 1/0.8 + 0.001*N*L
max_iter = 200



sum_EE = []
sum_rate = []
for n_iter in range(max_iter):
    # generate channel --------------------------------------------------
    LOS_bs_ris = gen_LOS(N,M,10,L)
    H, channel_bs_user, perfect_H, perfect_bs_user = generate_channel(M,N,L,K,1,LOS_bs_ris,0)
    H = H[0]
    channel_bs_user = channel_bs_user[0]
    # initialization ------------------------------------------------------ 
    W = np.random.uniform(0,1,size=(M,K+1))
    norm = np.linalg.norm(W,axis=0)
    W = (W/norm).transpose()*Pmax/np.sqrt(K+1)
    exp_random = np.exp(1j*np.random.uniform(0,np.pi,size=(W.shape)))
    W = W*exp_random
    Theta = np.exp(1j*np.random.uniform(0,np.pi,size=(L*N)))



    pre_gamma = np.random.uniform(0,1,K)
    pre_zeta = np.random.uniform(0,1,K)
    pre_lamb = np.random.uniform(0,1,K)
    pre_curv = np.random.uniform(0,1,K)
    pre_a = np.zeros(K)

    pre_beta = np.random.uniform(0,1,K)
    pre_eta = np.random.uniform(0,1,K)

    t = np.zeros((K,K+1,L*N),dtype=np.complex128)
    g = np.zeros((K,K+1),dtype=np.complex128)

    s = cp.Variable(L*N,complex=True)
    eta = cp.Variable(K,pos=True)
    beta = cp.Variable(K,pos=True)

    w = cp.Variable((K+1,M),complex=True)
    a = cp.Variable(K,pos=True)
    zeta = cp.Variable(K,pos=True)
    curv = cp.Variable(K,pos=True)
    gamma = cp.Variable(K,pos=True)
    lamb  = cp.Variable(K,pos=True)

    EE = 0
    rate = 0
    try:
        for theta_iter in range(5):
            # SCA -----------------------------------------------------------------------------------
            for k in range(K):
                for i in range(K+1):
                    concat_H = H[:,:,:,k].reshape((M,-1)).transpose()
                    t[k,i,:] = np.dot(concat_H,W[i,:].reshape((M,1))).squeeze()
                    g[k,i] = np.dot(channel_bs_user[k].reshape((1,-1)),W[i].reshape((-1,1))).squeeze()
            R0 = 2**np.sum(pre_a)-1
            # R0 = 0

            # Construct the problem.
            

            objective = cp.Maximize(cp.sum(cp.log1p(eta)) + 2*C*cp.sum(cp.real(cp.multiply(cp.conj(Theta),s-Theta))))

            constraints = []
            for k1 in range(K):
                interference = 0
                for k2 in range(1,K+1):
                    if k2-1 != k1:
                        interference += cp.square(cp.abs((t[k1,k2,:]@s) + g[k1,k2]))
                interference += sigma
                constraints += [interference<=beta[k1]]

            for k in range(K):
                LHS = 2*cp.real(cp.conj((t[k,k+1,:]@Theta)+g[k,k+1])*(t[k,k+1,:]@s))-cp.square(cp.abs((t[k,k+1,:]@Theta)+g[k,k+1]))
                RHS = 1/4*(cp.square(beta[k]+eta[k])-(pre_beta[k]-pre_eta[k])*(beta[k]-eta[k])+cp.square(pre_beta[k]-pre_eta[k]))
                constraints += [LHS >= RHS]

            for k in range(K):
                LHS = 2*cp.real(cp.conj((t[k,0,:]@Theta)+g[k,0])*(t[k,0,:]@s))-cp.square(cp.abs((t[k,0,:]@Theta)+g[k,0]))
                RHS = 0
                for i in range(1,K+1):
                    RHS += cp.square(cp.abs((t[k,i,:]@s)+g[k,i]))
                RHS = R0*(RHS + sigma)
                constraints += [LHS >= RHS]
            constraints += [beta >= 0]
            constraints += [eta >= 0]
            for i in range(L*N):
                constraints += [cp.abs(s[i]) <= 1]

            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.MOSEK)
            if objective.value != None:
                Theta = np.copy(s.value)
                pre_beta = np.copy(beta.value)
                pre_eta = np.copy(eta.value)
            else:
                Theta = Theta + np.exp(1j*np.random.uniform(0,np.pi,size=(L*N)))
                pre_beta = pre_beta + np.abs(np.random.normal(0,1,K))
                pre_eta = pre_eta + np.abs(np.random.normal(0,1,K))

            # print("objective1:",objective.value)
            # print("s:",Theta)
            # print("beta:",beta.value)
            # print("eta:",eta.value)

        for w_iter in range(15):
        # Dinkelbach -----------------------------------------------------------------------------------


            constraints = []

            til_g = np.zeros((K,M),dtype=np.complex128)
            for k in range(K):
                concat_H = H[:,:,:,k].reshape((M,-1))
                til_g[k,:] = np.dot(concat_H,Theta.reshape((-1,1))).squeeze() + channel_bs_user[k]
                

            for k1 in range(K):
                interference = 0
                for k2 in range(1,K+1):
                    if k2-1 != k1:
                        interference += cp.square(cp.abs((til_g[k1,:]@w[k2,:])))
                interference += sigma
                constraints += [interference <= gamma[k1]]

            for k in range(K):
                LHS = cp.real((til_g[k,:]@w[k+1,:]))
                RHS = cp.sqrt(pre_zeta[k]*pre_gamma[k])+1/2*cp.sqrt(pre_gamma[k]/pre_zeta[k])*(zeta[k]-pre_zeta[k])+1/2*cp.sqrt(pre_zeta[k]/pre_gamma[k])*(gamma[k]-pre_gamma[k])
                if RHS:
                    RHS = RHS
                else:
                    RHS = 0
                constraints += [LHS >= RHS]

            for k1 in range(K):
                interference = 0
                for k2 in range(1,K+1):
                    interference += cp.square(cp.abs((til_g[k1,:]@w[k2,:])))
                interference += sigma
                constraints += [interference <= lamb[k1]]

            for k in range(K):
                LHS = 2*cp.real((til_g[k]@W[0,:])*(til_g[k]@(w[0,:]))) - cp.square(cp.abs((til_g[k]@W[0,:])))
                RHS = 1/4*(cp.square(lamb[k]+curv[k])-(pre_lamb[k]-pre_curv[k])*(lamb[k]-curv[k])+cp.square(pre_lamb[k]-pre_curv[k]))
                constraints += [LHS >= RHS]

            constraints += [cp.sum(a) - cp.log1p(curv)/math.log(2) <= 0]

            constraints += [cp.sum_squares(w) <= (Pmax)]
            for i in range(K):
                constraints += [zeta[k] - (cp.exp(cp.multiply(np.log(2),(Rk-a[k])))-1) >= 0]
            constraints += [a >= 0]
            constraints += [gamma >= 0]
            constraints += [curv >=0]
            constraints += [lamb >= 0]
            constraints += [zeta >= 0]
            

            mu = np.sum(pre_a+np.log(1+pre_zeta)/math.log(2)) / (np.sum(np.abs(W)**2)+Pc)

            objective = cp.Maximize(cp.sum(a+cp.log1p(zeta)/math.log(2)) - mu*(cp.sum_squares(w)+Pc))
            problem = cp.Problem(objective,constraints)
            problem.solve(solver=cp.MOSEK)

            if objective != None:
                pre_a = np.copy(a.value)
                pre_curv = np.copy(curv.value)
                pre_gamma = np.copy(gamma.value)
                pre_zeta = np.copy(zeta.value)
                pre_lamb = np.copy(lamb.value)
                W = np.copy(w.value)
            else:
                pre_a = pre_a + np.abs(np.random.normal(0,1,K))
                pre_curv = pre_curv + np.abs(np.random.normal(0,1,K))
                pre_gamma = pre_gamma + np.abs(np.random.normal(0,1,K))
                pre_zeta = pre_zeta + np.abs(np.random.normal(0,1,K))
                pre_lamb = pre_lamb + np.abs(np.random.normal(0,1,K))
            # pre_zeta[pre_zeta==0] = 1e-7
            # pre_gamma[pre_gamma==0] = 1e-7
            # print("objective2:",objective.value)
            # print("a:",a.value)
            # print("curv:",curv.value)
            # print("gamma:",gamma.value)
            # print("zeta:",zeta.value)
            # print("lamb:",lamb.value)
            # print("W",w.value)    
        


            t = np.zeros((K,K+1,L*N),dtype=np.complex128)
            g = np.zeros((K,K+1),dtype=np.complex128)
            for k in range(K):
                for i in range(K+1):
                    concat_H = H[:,:,:,k].reshape((M,-1)).transpose()
                    t[k,i,:] = np.dot(concat_H,W[i,:].reshape((M,1))).squeeze()
                    g[k,i] = np.dot(channel_bs_user[k].reshape((1,-1)),W[i].reshape((-1,1))).squeeze()

            private_rate = 0
            theta = np.array(Theta)
            for k1 in range(K):
                interference = 0
                for k2 in range(1,K+1):
                    if k2-1 != k1:
                        interference += np.abs(np.dot(t[k1,k2,:],theta) + g[k1,k2])**2
                    else:
                        signal = np.abs(np.dot(t[k1,k2,:],theta) + g[k1,k2])**2
                interference += sigma
                private_rate += np.log(1+signal/interference)/math.log(2)
            a_ = np.array(a.value)
            W_ = np.array(W)
            rate = (np.sum(a_) + private_rate)
            EE = (np.sum(a_) + private_rate)/(np.sum(np.abs(W)**2)+Pc)

            print("EE:",EE,"bps/Hz/Joule")
        if EE > 10:
            sum_EE.append(EE)
            sum_rate.append(rate)
    except:
        if EE > 10:
            sum_EE.append(EE)
            sum_rate.append(rate)
sum_EE = np.array(sum_EE)
sum_rate = np.array(sum_rate)
print("EE:",np.mean(sum_EE),", Sum rate:",np.mean(sum_rate))


