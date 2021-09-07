#!/usr/bin/env python
# coding: utf-8

# In[120]:


import numpy as np
import scipy.stats as si
from scipy.sparse import diags
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt

# Market and option parameters
K, r, sigma,sigma_i, T  = 90, 0.05, 0.4, 0.2,1
mu = 0.2 #np.linspace(-1,1,101)


# In[121]:


# Grid parameters
s_min, s_max = 10, 150
N, M = 1000, 100  # N = 1000, M = 100

# Setup of grids, assume t0 = 0
dt = T/N
dx = (s_max - s_min)/M
s = np.linspace(s_min, s_max, M+1)
t = np.linspace(0, T, N+1)


# In[126]:


#hedging with actual volatility
def euro_vanilla_call(S, K, T, r, sigma):
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call

s=100
#hedging with actual volatility
P_ac= euro_vanilla_call(100, 90, T, 0.05, 0.4)-euro_vanilla_call(100, 90, T, 0.05, 0.2)
P_ac


# In[127]:


K = np.linspace(80,110,31)
P_ac = np.zeros(31)

for i in range(31):
    P_ac[i]= euro_vanilla_call(100, K[i], T, r, 0.4)-euro_vanilla_call(100, K[i], T, r, 0.2)


fig, ax = plt.subplots()
ax.plot(K, P_ac, 'k-', label="actual vol")
ax.set(xlabel='$K$', ylabel='indifference price', title=" indifference price vs K")
ax.legend()


# In[4]:


# Tridiagional matrix solver. a, b, c are the low, mid and high diagional of the matrix
# d is the constant vector on the RHS
#(copied shamelessly from https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9)
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


# In[5]:


# Set up the vectors A and C as they are constant wrt time
A = 0.5 * sigma**2 * dt / dx**2 * s**2 - 0.5 * mu * dt/ dx * s
C = 0.5 * sigma**2 * dt / dx**2 * s**2 + 0.5 * mu * dt/ dx * s

a_diag = np.concatenate([A[1:-1],[0]])
c_diag = np.concatenate([[0],C[1:-1]])


# In[6]:


F_im=np.ones(M+1)
u=-0.0001
for n in range(1,N+1):
    d1 = (np.log(s/K) + (r + 0.5 *sigma_i**2)*dt*n)/(sigma_i*np.sqrt(dt*n))
    Du = dt *u *(sigma**2 - sigma_i**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
    Dd =  2*sigma_i*np.sqrt(dt*n)*np.sqrt(2*np.pi)
    D = Du/Dd
    B = - sigma**2 * (dt / (dx**2)) * s**2 +  D
    b_diag = np.concatenate([[0],B[1:-1],[0]])
    d = F_im # The RHS of the system of equations is V^{n-1}
    F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
F1 = F_im[64]

F_im=np.ones(M+1)
u=0.0001
for n in range(1,N+1):
    d1 = (np.log(s/K) + (r + 0.5 *sigma_i**2)*dt*n)/(sigma_i*np.sqrt(dt*n))
    Du = dt *u *(sigma**2 - sigma_i**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
    Dd =  2*sigma_i*np.sqrt(dt*n)*np.sqrt(2*np.pi)
    D = Du/Dd
    B = - sigma**2 *( dt / (dx**2) )* s**2 +  D
    b_diag = np.concatenate([[0],B[1:-1],[0]])
    d = F_im # The RHS of the system of equations is V^{n-1}
    F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
F2 = F_im[64]

F_im=np.ones(M+1)
u=0
for n in range(1,N+1):
    d1 = (np.log(s/K) + (r + 0.5 *sigma_i**2)*dt*n)/(sigma_i*np.sqrt(dt*n))
    Du = dt *u *(sigma**2 - sigma_i**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
    Dd =  2*sigma_i*np.sqrt(dt*n)*np.sqrt(2*np.pi)
    D = Du/Dd
    B = - sigma**2 * (dt / (dx**2)) * s**2 +  D
    b_diag = np.concatenate([[0],B[1:-1],[0]])
    d = F_im # The RHS of the system of equations is V^{n-1}
    F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
F3 = F_im[64]

mean = (F2-F1)/0.0002
I2 = ((F2-F3)/0.0001-(F3-F1)/0.0001)/0.0001


# In[11]:


mean


# In[12]:


mean
std = np.sqrt(I2-mean**2)
std


# 

# In[79]:


#initializing for different mu
mu = np.linspace(-1,1,101)
K=100
P = np.zeros(101)
SD = np.zeros(101)
SK =  np.zeros(101)
KT =  np.zeros(101)

for m in range(101):
    A = 0.5 * sigma**2 * dt / dx**2 * s**2 - 0.5 * mu[m] * dt/ dx * s
    C = 0.5 * sigma**2 * dt / dx**2 * s**2 + 0.5 * mu[m] * dt/ dx * s

    a_diag = np.concatenate([A[1:-1],[0]])
    c_diag = np.concatenate([[0],C[1:-1]])
    
    F_im = np.ones(M+1)

    u=-0.0001
    for n in range(1,N+1):
        d1 = (np.log(s/K) + (r + 0.5 *sigma_i**2)*dt*n)/(sigma_i*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
    F1 = F_im[64]

    F_im=np.ones(M+1)
    u=0.0001
    for n in range(1,N+1):
        d1 = (np.log(s/K) + (r + 0.5 *sigma_i**2)*dt*n)/(sigma_i*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
    F2 = F_im[64]
    
    F_im=np.ones(M+1)
    u=0
    for n in range(1,N+1):
        d1 = (np.log(s/K) + (r + 0.5 *sigma_i**2)*dt*n)/(sigma_i*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
    F3 = F_im[64]
    
    F_im=np.ones(M+1)
    u=-0.0002
    for n in range(1,N+1):
        d1 = (np.log(s/K) + (r + 0.5 *sigma_i**2)*dt*n)/(sigma_i*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    F4 = F_im[64]
    
    F_im=np.ones(M+1)
    u=0.0002
    for n in range(1,N+1):
        d1 = (np.log(s/K) + (r + 0.5 *sigma_i**2)*dt*n)/(sigma_i*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    F5 = F_im[64]
    
    F_im=np.ones(M+1)
    u=0.0003
    for n in range(1,N+1):
        d1 = (np.log(s/K) + (r + 0.5 *sigma_i**2)*dt*n)/(sigma_i*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    F6 = F_im[64]
    
    F_im=np.ones(M+1)
    u=-0.0003
    for n in range(1,N+1):
        d1 = (np.log(s/K) + (r + 0.5 *sigma_i**2)*dt*n)/(sigma_i*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    F7 = F_im[64]

    mean = (F2-F1)/0.0002
    I2 = ((F2-F3)/0.0001-(F3-F1)/0.0001)/0.0001
    sd = np.sqrt(I2-mean**2)
    I3 = ((((F5-F2)/0.0001 - (F2 - F3)/0.0001)/0.0001) - ((F3-F1)/0.0001-(F1 - F4)/0.0001)/0.0001)/0.0001
    I4 = ((((F6-F5)/0.0001 - (F5-F2)/0.0001)/0.0001 - ((F5 - F2)/0.0001 -(F2 - F3)/0.0001)/0.0001)/0.0001-(((F3-F1)/0.0001 - (F1-F4)/0.0001)/0.0001 - ((F1 - F4)/0.0001 -(F4 - F7)/0.0001)/0.0001)/0.0001)/0.0001
    sk = I3 - 3*mean*I2+ 2*mean**3
    kt = I4 - 4*mean*I3 + 6*mean**2 * I2 - 3*mean**4
    
    P[m]=mean
    SD[m]=sd
    SK[m]=sk
    KT[m]=kt


# In[68]:


P_ac= euro_vanilla_call(s, K, T, r, sigma)-euro_vanilla_call(s, K, T, r, sigma_i) 


# In[69]:


mu = np.linspace(-1,1,101)
fig, ax = plt.subplots()
ax.plot(mu, P, 'k-', label="implied vol")
ax.plot(mu, P_ac[64]*np.ones(101), '.', linewidth=0.5, label="actual vol")
ax.set(xlabel='$\mu$', ylabel='Expected profit', title="mean vs $\mu$ (K=100)")
ax.legend()


# In[70]:


mu = np.linspace(-1,1,101)
fig, ax = plt.subplots()
ax.plot(mu, P, 'r-', label="expected profit")
ax.plot(mu, SD, 'g-', label="standard deviation")
plt.axhline(0, color='black')
plt.axvline(0, color='black')
ax.set(xlabel='$\mu$', ylabel='Expected profit and sd', title="mean and sd vs $\mu$ (K=100)")
ax.legend()


# In[80]:


mu = np.linspace(-1,1,101)
fig, ax = plt.subplots()
ax.plot(mu, SK, 'r-', label="skewness")
ax.plot(mu, KT, 'g-', label="kurtosis")
plt.axhline(0, color='black')
plt.axvline(0, color='black')
ax.set(xlabel='$\mu$', ylabel='skewness and kurtosis', title="skewness and kurtosis vs $\mu$ (K=100)")
ax.legend()


# In[104]:


#initializing for different K
K = np.linspace(80,120,41)
P = np.zeros(41)
SD = np.zeros(41)
SK = np.zeros(41)
KT = np.zeros(41)
mu = 0.2

for m in range(41):
    A = 0.5 * sigma**2 * dt / dx**2 * s**2 - 0.5 * mu * dt/ dx * s
    C = 0.5 * sigma**2 * dt / dx**2 * s**2 + 0.5 * mu * dt/ dx * s

    a_diag = np.concatenate([A[1:-1],[0]])
    c_diag = np.concatenate([[0],C[1:-1]])
    
    F_im = np.ones(M+1)

    u=-0.0001
    for n in range(1,N+1):
        d1 = (np.log(s/K[m]) + (r + 0.5 *sigma_i**2)*dt*n)/(sigma_i*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
    F1 = F_im[64]

    F_im=np.ones(M+1)
    u=0.0001
    for n in range(1,N+1):
        d1 = (np.log(s/K[m]) + (r + 0.5 *sigma_i**2)*dt*n)/(sigma_i*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
    F2 = F_im[64]
    
    F_im=np.ones(M+1)
    u=0
    for n in range(1,N+1):
        d1 = (np.log(s/K[m]) + (r + 0.5 *sigma_i**2)*dt*n)/(sigma_i*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
    F3 = F_im[64]
    
    F_im=np.ones(M+1)
    u=-0.0002
    for n in range(1,N+1):
        d1 = (np.log(s/K[m]) + (r + 0.5 *sigma_i**2)*dt*n)/(sigma_i*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
    F4 = F_im[64]
    
    

    F_im=np.ones(M+1)
    u=0.0002
    for n in range(1,N+1):
        d1 = (np.log(s/K[m]) + (r + 0.5 *sigma_i**2)*dt*n)/(sigma_i*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
    F5 = F_im[64]
    
    
    F_im=np.ones(M+1)
    u=0.0003
    for n in range(1,N+1):
        d1 = (np.log(s/K[m]) + (r + 0.5 *sigma_i**2)*dt*n)/(sigma_i*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
    F6 = F_im[64]
    
    F_im=np.ones(M+1)
    u=-0.0003
    for n in range(1,N+1):
        d1 = (np.log(s/K[m]) + (r + 0.5 *sigma_i**2)*dt*n)/(sigma_i*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
    F7 = F_im[64]
    
    
    
    mean = (F2-F1)/0.0002
    I2 = ((F2-F3)/0.0001-(F3-F1)/0.0001)/0.0001
    sd = np.sqrt(I2-mean**2)
    I3 = ((((F5-F2)/0.0001 - (F2 - F3)/0.0001)/0.0001) - ((F3-F1)/0.0001-(F1 - F4)/0.0001)/0.0001)/0.0001
    I4 = ((((F6-F5)/0.0001 - (F5-F2)/0.0001)/0.0001 - ((F5 - F2)/0.0001 -(F2 - F3)/0.0001)/0.0001)/0.0001-(((F3-F1)/0.0001 - (F1-F4)/0.0001)/0.0001 - ((F1 - F4)/0.0001 -(F4 - F7)/0.0001)/0.0001)/0.0001)/0.0001
    sk = I3 - 3*mean*I2+ 2*mean**3
    kt = I4 - 4*mean*I3 + 6*mean**2 * I2 - 3*mean**4
    
    P[m]=mean
    SD[m]=sd
    SK[m]=sk
    KT[m]=kt


# In[97]:


q = np.linspace(80,120,41)


# In[105]:


P_acK = np.zeros(41)
K = list(np.linspace(80,120,41))
for i in range(41):
    profit = euro_vanilla_call(s, K[i], T, r, sigma)-euro_vanilla_call(s, K[i], T, r, sigma_i)
    P_acK[i] = profit[64]


# In[106]:


K = np.linspace(80,120,41)
fig, ax = plt.subplots()
ax.plot(K, P, 'k-', label="implied")
ax.plot(K, P_acK, '.', linewidth=0.5, label="actual")
ax.set(xlabel='$K$', ylabel='expected profit', title="expected profit vs K ($\mu$ = 0.2)")
ax.legend()


# In[107]:


K = np.linspace(80,120,41)
fig, ax = plt.subplots()
ax.plot(K, P, 'k-', label="Expected profit")
ax.plot(K, SD, 'g-', label="Standard deviation")
ax.set(xlabel='$K$', ylabel='mean and standard deviation', title="mean and standard deviation vs K ($\mu$ = 0.2)")
ax.legend()


# In[108]:


K = np.linspace(80,120,41)
fig, ax = plt.subplots()
ax.plot(K, SK, 'k-', label="skewness")
ax.plot(K, KT, 'g-', label="kurtosis")
plt.axhline(0, color='black')

ax.set(xlabel='$K$', ylabel='skewness and kurtosis', title="skewness and kurtosis vs K ($\mu$ = 0.2)")
ax.legend()


# In[109]:


#changing the difference between implied vol and actual vol
K, r, sigma, T  = 90, 0.05, 0.4,1
mu = 0.2
sigma_i = np.linspace(0.2,0.6,41)
P = np.zeros(41)
SD = np.zeros(41)
SK = np.zeros(41)
KT = np.zeros(41)


for m in range(41):
    A = 0.5 * sigma**2 * dt / dx**2 * s**2 - 0.5 * mu * dt/ dx * s
    C = 0.5 * sigma**2 * dt / dx**2 * s**2 + 0.5 * mu * dt/ dx * s

    a_diag = np.concatenate([A[1:-1],[0]])
    c_diag = np.concatenate([[0],C[1:-1]])
    
    F_im = np.ones(M+1)

    u=-0.0001
    for n in range(1,N+1):
        d1 = (np.log(s/K) + (r + 0.5 *sigma_i[m]**2)*dt*n)/(sigma_i[m]*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i[m]**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i[m]*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
    F1 = F_im[64]

    F_im=np.ones(M+1)
    u=0.0001
    for n in range(1,N+1):
        d1 = (np.log(s/K) + (r + 0.5 *sigma_i[m]**2)*dt*n)/(sigma_i[m]*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i[m]**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i[m]*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
    F2 = F_im[64]
    
    F_im=np.ones(M+1)
    u=0
    for n in range(1,N+1):
        d1 = (np.log(s/K) + (r + 0.5 *sigma_i[m]**2)*dt*n)/(sigma_i[m]*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i[m]**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i[m]*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
    F3 = F_im[64]
    
    F_im=np.ones(M+1)
    u=-0.0002
    for n in range(1,N+1):
        d1 = (np.log(s/K) + (r + 0.5 *sigma_i[m]**2)*dt*n)/(sigma_i[m]*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i[m]**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i[m]*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
    F4 = F_im[64]
    
    

    F_im=np.ones(M+1)
    u=0.0002
    for n in range(1,N+1):
        d1 = (np.log(s/K) + (r + 0.5 *sigma_i[m]**2)*dt*n)/(sigma_i[m]*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i[m]**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i[m]*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
    F5 = F_im[64]
    
    
    F_im=np.ones(M+1)
    u=0.0003
    for n in range(1,N+1):
        d1 = (np.log(s/K) + (r + 0.5 *sigma_i[m]**2)*dt*n)/(sigma_i[m]*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i[m]**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i[m]*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
    F6 = F_im[64]
    
    F_im=np.ones(M+1)
    u=-0.0003
    for n in range(1,N+1):
        d1 = (np.log(s/K) + (r + 0.5 *sigma_i[m]**2)*dt*n)/(sigma_i[m]*np.sqrt(dt*n))
        Du = dt *u *(sigma**2 - sigma_i[m]**2)*np.exp(-r*(T-dt*n))*s*np.exp(-0.5*d1**2)
        Dd =  2*sigma_i[m]*np.sqrt(dt*n)*np.sqrt(2*np.pi)
        D = Du/Dd
        B = - sigma**2 * dt / dx**2 * s**2 +  D
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        d = F_im # The RHS of the system of equations is V^{n-1}
        F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
    
    F7 = F_im[64]
    
    
    
    mean = (F2-F1)/0.0002
    I2 = ((F2-F3)/0.0001-(F3-F1)/0.0001)/0.0001
    sd = np.sqrt(I2-mean**2)
    I3 = ((((F5-F2)/0.0001 - (F2 - F3)/0.0001)/0.0001) - ((F3-F1)/0.0001-(F1 - F4)/0.0001)/0.0001)/0.0001
    I4 = ((((F6-F5)/0.0001 - (F5-F2)/0.0001)/0.0001 - ((F5 - F2)/0.0001 -(F2 - F3)/0.0001)/0.0001)/0.0001-(((F3-F1)/0.0001 - (F1-F4)/0.0001)/0.0001 - ((F1 - F4)/0.0001 -(F4 - F7)/0.0001)/0.0001)/0.0001)/0.0001
    sk = I3 - 3*mean*I2+ 2*mean**3
    kt = I4 - 4*mean*I3 + 6*mean**2 * I2 - 3*mean**4
    
    P[m]=mean
    SD[m]=sd
    SK[m]=sk
    KT[m]=kt


# In[113]:


P_acK = np.zeros(41)
sigma_i= list(np.linspace(0.2,0.6,41))
for i in range(41):
    profit = euro_vanilla_call(s, K, T, r, sigma)-euro_vanilla_call(s, K, T, r, sigma_i[i])
    P_acK[i] = profit[64]


# In[114]:


sigma_i = np.linspace(0.2,0.6,41)
fig, ax = plt.subplots()
ax.plot(sigma_i, P, 'k-', label="implied")
ax.plot(sigma_i, P_acK, '.', linewidth=0.5, label="actual")
ax.set(xlabel='sigma_i', ylabel='expected profit', title="expected profit vs sigma_i ")
ax.legend()


# In[115]:


sigma_i = np.linspace(0.2,0.6,41)
fig, ax = plt.subplots()
ax.plot(sigma_i, P, 'k-', label="Expected profit")
ax.plot(sigma_i, SD, 'g-', label="Standard deviation")
ax.set(xlabel='sigma_i', ylabel='mean and standard deviation', title="mean and standard deviation vs sigma_i")
ax.legend()


# In[116]:


sigma_i = np.linspace(0.2,0.6,41)
fig, ax = plt.subplots()
ax.plot(sigma_i, SK, 'k-', label="skewness")
ax.plot(sigma_i, KT, 'g-', label="kurtosis")
plt.axhline(0, color='black')

ax.set(xlabel='sigma_i', ylabel='skewness and kurtosis', title="skewness and kurtosis vs sigma_i")
ax.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


#3D graph
r, sigma,sigma_i, T,D  = 0.05, 0.4, 0.2,1 , 0

mu = np.linspace(-0.2,0.2,11)
K = np.linspace(80,120,41)

# Grid parameters
s_min, s_max = 10, 150
N, M = 1000, 100  # N = 1000, M = 100

# Setup of grids, assume t0 = 0
dt = T/N
dx = (s_max - s_min)/M
s = np.linspace(s_min, s_max, M+1)
t = np.linspace(0, T, N+1)

#initializing
P = np.asarray([[0]*11]*41,dtype='int64')

for o in range(41):
    for m in range(11):
        A = 0.5 * sigma**2 * dt / dx**2 * s**2 - 0.5 * mu[m] * dt/ dx * s
        B = - sigma**2 * dt / dx**2 * s**2 
        C = 0.5 * sigma**2 * dt / dx**2 * s**2 + 0.5 * mu[m] * dt/ dx * s

         # Setup the matrix L and I
        a_diag = np.concatenate([A[1:-1],[0]])
        b_diag = np.concatenate([[0],B[1:-1],[0]])
        c_diag = np.concatenate([[0],C[1:-1]])
        L = diags([a_diag, b_diag, c_diag], [-1, 0, 1]).toarray()
        I = np.identity(M+1)
    
        F_im = np.ones(M+1)
        u = 1

        for n in range(1,N+1):
            d2 = (np.log(s/K[o]) + (r - 0.5 *sigma_i**2)*dt*n)/(sigma_i*np.sqrt(dt*n))
            Du = dt *u* K[o] *(sigma**2 - sigma_i**2)*np.exp(-r)*np.exp(-0.5*d2**2)
            Dd =  2*sigma_i*np.sqrt(2*np.pi*dt*n)
            D = Du/Dd
            d = F_im +D  # The RHS of the system of equations is V^{n-1}
            F_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n
        P[o,m] = F_im[64]  

P


# In[129]:


# Black-Scholes call option formula
K = 110
def euro_vanilla_call(S, K, T, r, sigma):
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call

#hedging with actual volatility
P_ac= euro_vanilla_call(s, K, T, r, sigma)-euro_vanilla_call(s, K, T, r, sigma_i)
P_ac


# In[130]:


Profit_ac=(P_ac[64]+0.4*P_ac[65])/1.4


# In[132]:


Profit_ac


# In[45]:


P_ac[64]


# In[ ]:




