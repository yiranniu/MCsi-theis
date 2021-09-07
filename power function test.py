#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.stats as si
from scipy.sparse import diags
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d


# In[6]:


def MGF_2D_FD_ADI(K, mu, r, sigma, sigma_i, T,  s_min, s_max, I_min, I_max, M, N, J):

    # Setup of grids
    dt = T/N
    dx = (s_max - s_min)/M
    di = (I_max - I_min)/J
    s = np.linspace(s_min, s_max, M+1)
    I = np.linspace(I_min, I_max, J+1)
    t = np.linspace(0, T, N+1)

    # Set up the elements in the tridiagonal matrix in the first half step
    A = 0.25 * sigma**2 * dt / dx**2 * s**2 - 0.25 * mu * dt/ dx * s
    B = -0.5* sigma**2 * dt / dx**2 * s**2
    C = 0.25 * sigma**2 * dt / dx**2 * s**2 + 0.25 * mu * dt/ dx * s
    a_diag = np.concatenate([A[1:-1],[0]])
    b_diag = np.concatenate([[0],B[1:-1],[0]])
    c_diag = np.concatenate([[0],C[1:-1]]) 
  
    F_im = np.zeros((M+1,J+1))
    # terminal condition: F(S,I,T)=e^{uI} <=> F^0_{k,j}=e^{u I_j}
    for j in range(J+1):
        F_im[:,j] = (w0 +I[j])**alpha 

    # Loop in time
    for n in range(N):

        # Display the progress of the loop in time
        if n % 50 == 0:
            print('Time loop: %d of %s done' % (n, N))

        # implicit scheme in s-direction, explicit scheme in I-direction
        for j in range(J+1):

            # Set the RHS of the matrix equations
            if j==0:
                d = F_im[:,j] + 0.5 * dt / di * g_fun(T - n*dt, s, K, r, sigma_i, sigma, T) * (F_im[:,j+1] - F_im[:,j])
            elif j == J:
                d = F_im[:,j] + 0.5 * dt / di * g_fun(T - n*dt, s, K, r, sigma_i, sigma, T) * (F_im[:,j] - F_im[:,j-1])
            else:
                d = F_im[:,j] + 0.25 * dt / di * g_fun(T - n*dt, s, K, r, sigma_i, sigma, T) * (F_im[:,j+1] - F_im[:,j-1])

            d[0] = (w0 +I[j])**alpha #boundary condition at s_min
            d[M] = (w0 +I[j])**alpha  #boundary condition at s_max
            F_im[:,j] = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d) # get the value of v^{n+1/2}_{kj} for each k,j

        # implicit scheme in I-direction, explicit scheme in S-direction
        for k in range(1,M): #no need to consider k=0 and k=M which will be given by the boundary conds
            
            # Set up the elements in the tridiagonal matrix in the second half step
            D_val = -0.25 * g_fun(T - (n+1)*dt, s[k], K, r, sigma_i, sigma, T) * dt / di
            D = D_val * np.ones(J+1)
            E = -D
            d_diag = np.concatenate([D[1:-1],[2*D_val]])
            e_diag = np.concatenate([[-2*D_val],E[1:-1]]) 
            Q = np.ones(J+1)
            q_diag = np.concatenate([[1-2*D_val],Q[1:-1],[1+2*D_val]])

            # Set the RHS of the matrix equations
            d = F_im[k,:] + 0.25 * sigma**2 * dt / dx**2 * s[k]**2 * (F_im[k-1,:]-2*F_im[k,:]+F_im[k+1,:]) + 0.25 * mu * dt/ dx * s[k] * (F_im[k+1,:]-F_im[k-1,:])
            F_im[k,:] = TDMAsolver(-d_diag, q_diag, -e_diag, d) # get the value of v^{n+1}_{kj} for each k,j
        
        # boundary conds at k=0 and k=M (s_min and s_max)
        F_im[0,:] = (w0 +I)**alpha 
        F_im[M,:] = (w0 +I)**alpha 

    return interp2d(I, s, F_im, kind='cubic')


# In[7]:


def BS_Gamma(t, s, K, r, sigma_i, T):

    if t == T:
        return 0
    else:
        d1 = (np.log(s/K) + (r + 0.5 * sigma_i **2) * (T-t)) / (sigma_i * np.sqrt(T-t))
        return (np.exp(-0.5 * d1**2) / np.sqrt(2*np.pi)) / (s * sigma_i *np.sqrt(T-t))
    
def g_fun(t, s, K, r, sigma_i, sigma, T):
    return 0.5 * (sigma**2 - sigma_i**2) * np.exp(-r * t) * s**2 * BS_Gamma(t, s, K, r, sigma_i, T)

def TDMAsolver(a, b, c, d):

    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc


# In[8]:


# Market and option parameters
#def MGF_2D_FD_ADI(K, mu, r, sigma, sigma_i, T, u, s_min, s_max, I_min, I_max, M, N, J):
K, r, sigma,sigma_i, T  = 90, 0.05, 0.4, 0.2,1
mu = 0.3
alpha = 0.5
#initial wealth
w0 = 1
s_min, s_max = 10, 150
I_min, I_max, J = 0, 50, 500
N, M = 1000, 100 
s =np.linspace(10,150,100)


# In[31]:


def euro_vanilla_call(S, K, T, r, sigma):
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call


# In[30]:


value =np.zeros(11)
mu = np.linspace(-0.5,0.5,11)

for i in range(11):
    test = MGF_2D_FD_ADI(K, mu[i], r, sigma, sigma_i, T,  s_min, s_max, I_min, I_max, M, N, J)
    value[i] = test(0,64)

c = value**2-1
fig, ax = plt.subplots()
ax.plot(mu, c, 'g-', label="power utility")
ax.set(xlabel='$\mu$', ylabel='indiferrence value', title="c vs $\mu$")
ax.legend()


# In[ ]:


c = value**2-1


# In[21]:


fig, ax = plt.subplots()
ax.plot(mu, c, 'g-', label="power utility")
ax.set(xlabel='$\mu$', ylabel='indiferrence value', title="c vs $\mu$")
ax.legend()


# In[32]:


#c vs K
mu, r, sigma,sigma_i, T  = 0.2, 0.05, 0.4, 0.2,1
alpha = 0.5
#initial wealth
w0 = 1
s_min, s_max = 10, 150
I_min, I_max, J = 0, 50, 500
N, M = 1000, 100 
s =np.linspace(10,150,100)
value =np.zeros(21)
K = np.linspace(80,100,21)

for i in range(21):
    test = MGF_2D_FD_ADI(K[i], mu, r, sigma, sigma_i, T,  s_min, s_max, I_min, I_max, M, N, J)
    value[i] = test(0,64)

c = value**2-1

P_ac = np.zeros(21)
for i in range(21):
    P_ac[i]= euro_vanilla_call(100, K[i], T, r, 0.4)-euro_vanilla_call(100, K[i], T, r, 0.2)


fig, ax = plt.subplots()
ax.plot(K, P_ac, 'k-', label="actual vol")
ax.plot(K, c, 'g-', label="implied&power utility")
ax.set(xlabel='$K$', ylabel='indifference price', title=" power utility indifference price vs K")
ax.legend()


# In[28]:


#c vs alpha
mu,K, r, sigma,sigma_i, T  = 0.2,90, 0.05, 0.4, 0.2,1
#initial wealth
w0 = 1
s_min, s_max = 10, 150
I_min, I_max, J = 0, 50, 500
N, M = 1000, 100 
s =np.linspace(10,150,100)
c =np.zeros(9)
A = np.linspace(0.1,0.9,9)

for i in range(9):
    alpha = A[i]
    alpha_i = 1/alpha
    test = MGF_2D_FD_ADI(K, mu, r, sigma, sigma_i, T,  s_min, s_max, I_min, I_max, M, N, J)
    
    c[i] = test(0,64)**(alpha_i)-1

fig, ax = plt.subplots()
ax.plot(A, c, 'g-', label="power utility")
ax.set(xlabel='alpha', ylabel='indiferrence value', title="c vs alpha")
ax.legend()

