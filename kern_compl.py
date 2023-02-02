import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import markers
from gd_algs import gd_alg
from time import sleep
import sys

def kern_gauss(X1,X2,sigma):
  if X1.shape[1]==1 and X2.shape[1]==1:
    return np.exp(-0.5*np.square((X1-X2.T)/sigma))
  X1X1=np.sum(np.square(X1),1).reshape((-1,1))
  X2X2=np.sum(np.square(X2),1).reshape((-1,1))
  X1X2=X1@X2.T
  D2=X1X1-2*X1X2+X2X2.T
  return np.exp(-0.5*D2/sigma**2)

def r2(y,y_hat):
  return 1-np.mean((y-y_hat)**2)/np.mean((y-np.mean(y))**2)

def mse(y,y_hat):
  return np.mean((y-y_hat)**2)

fig,axs=plt.subplots(1,1,figsize=(12,6))
fig1,axs1=plt.subplots(1,1,figsize=(12,6))

def f(x,n_per):
  return np.sin(n_per*2*np.pi*x)

step_size=0.1

sigma_0=2
sigma_m=1e-5
alpha=30
t=1e3

r2_tr=[]
r2_val=[]

sigmas_krr=np.array([1,0.3,0.1,0.03,0.01,0.003,0.001])
r2_tr_krr={}
r2_val_krr={}
for sigma_krr in sigmas_krr:
  r2_tr_krr[sigma_krr]=[]
  r2_val_krr[sigma_krr]=[]

n_val=200
n_pers=np.arange(1,10)
for n_per in n_pers:
  r2_tr_seed=[]
  r2_val_seed=[]
  r2_tr_seed_krr={}
  r2_val_seed_krr={}
  for sigma_krr in sigmas_krr:
    r2_tr_seed_krr[sigma_krr]=[]
    r2_val_seed_krr[sigma_krr]=[]
  for seed in range(10):
    np.random.seed(seed)
    n_tr=20*n_per
    x_tr=np.random.uniform(-1,1,n_tr).reshape((-1,1))
    y_tr=f(x_tr,n_per)+np.random.normal(0,.2,x_tr.shape)
    x_val=np.random.uniform(-1,1,n_val).reshape((-1,1))
    y_val=f(x_val,n_per)#+np.random.normal(0,0,x_val.shape)
    for sigma_krr in sigmas_krr:
      y1_tr_krr=kern_gauss(x_tr,x_tr,sigma_krr)@np.linalg.solve(kern_gauss(x_tr,x_tr,sigma_krr)+1/t*np.eye(n_tr),y_tr)
      y1_val_krr=kern_gauss(x_val,x_tr,sigma_krr)@np.linalg.solve(kern_gauss(x_tr,x_tr,sigma_krr)+1/t*np.eye(n_tr),y_tr)
      r2_tr_seed_krr[sigma_krr].append(r2(y_tr,y1_tr_krr))
      r2_val_seed_krr[sigma_krr].append(r2(y_val,y1_val_krr))
    
    xs=np.vstack((x_tr,x_val))
    xs_argsort=xs.argsort(0)
    y1=np.zeros(xs.shape)
    sigma=sigma_0
    for i in range(int(t/step_size)):
      Ks=kern_gauss(xs,x_tr,sigma)
      gd_obj=gd_alg(Ks,y_tr,'gd','pred',step_size, var0=y1)
      if np.max(np.abs(y1))<1000:
        gd_obj.gd_step()
      y1=gd_obj.get_fs()
      #sigma=(sigma+sigma_m*alpha*step_size)/(1+alpha*step_size)
      sigma=(sigma-sigma_m**2*alpha*step_size)/(1+alpha*step_size*(sigma-2*sigma_m))
    axs1.cla()
    axs1.plot(x_tr,y_tr,'ok')
    axs1.plot(x_val,y_val,'xr')
    axs1.plot(xs[xs_argsort,0],y1[xs_argsort,0])
    axs1.set_ylim([-2,2])
    fig1.savefig('figures/kern_compl_b.pdf')
    sleep(.1)
    print(seed,alpha,n_per,r2(y_tr,y1[:n_tr,:]),r2(y_val,y1[n_tr:,:]))
    r2_tr_seed.append(r2(y_tr,y1[:n_tr,:]))
    r2_val_seed.append(r2(y_val,y1[n_tr:,:]))
  
  axs.cla()
  r2_tr.append(r2_tr_seed)
  r2_val.append(r2_val_seed)
  r2_tr_mean=np.mean(np.array(r2_tr),1)
  r2_val_mean=np.mean(np.array(r2_val),1)
  #axs.plot(n_pers[:len(r2_tr)],r2_tr_mean)
  axs.plot(n_pers[:len(r2_val)],r2_val_mean,'C0')
  
  r2_tr_mean_krr={}
  r2_val_mean_krr={}
  for c,sigma_krr in enumerate(sigmas_krr):
    r2_tr_krr[sigma_krr].append(r2_tr_seed_krr[sigma_krr])
    r2_val_krr[sigma_krr].append(r2_val_seed_krr[sigma_krr])
    r2_tr_mean_krr[sigma_krr]=np.mean(np.array(r2_tr_krr[sigma_krr]),1)
    r2_val_mean_krr[sigma_krr]=np.mean(np.array(r2_val_krr[sigma_krr]),1)
    #axs.plot(n_pers[:len(r2_tr_krr[sigma_krr])],r2_tr_mean_krr[sigma_krr])
    axs.plot(n_pers[:len(r2_val_krr[sigma_krr])],r2_val_mean_krr[sigma_krr],'C'+str(c+1))
  axs.set_ylim([-0.01,1.01])
  fig.savefig('figures/kern_compl.pdf')
