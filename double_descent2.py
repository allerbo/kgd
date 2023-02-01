import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import markers
from gd_algs import gd_alg
from time import sleep
import sys
from sklearn.gaussian_process.kernels import Matern
from scipy.special import kv, gamma

def kern_gauss(X1,X2,sigma):
  if X1.shape[1]==1 and X2.shape[1]==1:
    return np.exp(-0.5*np.square((X1-X2.T)/sigma))
  X1X1=np.sum(np.square(X1),1).reshape((-1,1))
  X2X2=np.sum(np.square(X2),1).reshape((-1,1))
  X1X2=X1@X2.T
  D2=X1X1-2*X1X2+X2X2.T
  return np.exp(-0.5*D2/sigma**2)

def f(x):
  y=(x<-10)*(x+10)+(x>10)*(x-10)+(np.abs(x)<10)*20*np.sin(5*2*np.pi*x)
  #y=2*np.sin(2*np.pi*x)
  return y

def r2(y,y_hat):
  return 1-np.mean((y-y_hat)**2)/np.mean((y-np.mean(y))**2)

def mse(y,y_hat):
  return np.mean((y-y_hat)**2)


l=1
std=8

n_tr=100
n_val=100

fig,axs=plt.subplots(1,1,figsize=(12,6))
fig1,axs1=plt.subplots(1,1,figsize=(12,6))

step_size=0.1

alpha=3
sigma_ms=np.geomspace(10,0.0001,10)
t=1e3

r2_tr=[]
r2_val=[]
r2_tr_c=[]
r2_val_c=[]

for sigma_m in sigma_ms:
  r2_tr_seed=[]
  r2_val_seed=[]
  r2_tr_c_seed=[]
  r2_val_c_seed=[]
  for seed in range(10):
    np.random.seed(seed)
    x_tr=np.random.uniform(-20,20,n_tr).reshape((-1,1))
    #x_tr=np.random.normal(0,std,n_tr).reshape((-1,1))
    #x_tr=0.01*np.random.standard_cauchy(n_tr).reshape((-1,1))
    y_tr=f(x_tr)+np.random.normal(0,.2,x_tr.shape)
    #y_tr=f(x_tr)+0.05*np.random.standard_cauchy(x_tr.shape)
    #x_val=np.random.normal(0,std,n_val).reshape((-1,1))
    x_val=np.random.uniform(np.min(x_tr),np.max(x_tr),n_val).reshape((-1,1))
    y_val=f(x_val)+np.random.normal(0,0,x_val.shape)
    xs=np.vstack((x_tr,x_val))
    xs_argsort=xs.argsort(0)
     
    Ks_c=kern_gauss(xs,x_tr,sigma_m)
    gd_obj_c=gd_alg(Ks_c,y_tr,'gd','pred',step_size)
    y1=np.zeros(xs.shape)
    sigma=np.max(x_tr)-np.min(x_tr)
    for i in range(int(t/step_size)):
      if np.max(np.abs(gd_obj_c.get_fs()))<1000:
        gd_obj_c.gd_step()
      Ks=kern_gauss(xs,x_tr,sigma)
      gd_obj=gd_alg(Ks,y_tr,'gd','pred',step_size, var0=y1)
      if np.max(np.abs(y1))<1000:
        gd_obj.gd_step()
      y1=gd_obj.get_fs()
      #sigma=(sigma+sigma_m*alpha*step_size)/(1+alpha*step_size)
      sigma=(sigma-sigma_m**2*alpha*step_size)/(1+alpha*step_size*(sigma-2*sigma_m))
      #if i % 1000==0:
    #axs1.cla()
    #axs1.plot(x_tr,y_tr,'ok')
    #axs1.plot(x_val,y_val,'vk')
    #axs1.plot(xs[xs_argsort,0],y1[xs_argsort,0])
    #y1_c=gd_obj_c.get_fs()
    #axs1.plot(xs[xs_argsort,0],y1_c[xs_argsort,0])
    #fig1.savefig('figures/double_descent2b.pdf')
    #sleep(.1)
    y1_c=gd_obj_c.get_fs()
    print(alpha,sigma,r2(y_tr,y1[:n_tr,:]),r2(y_val,y1[n_tr:,:]))
    print('c',sigma_m,r2(y_tr,y1_c[:n_tr,:]),r2(y_val,y1_c[n_tr:,:]))
    r2_tr_seed.append(r2(y_tr,y1[:n_tr,:]))
    r2_val_seed.append(r2(y_val,y1[n_tr:,:]))
    r2_tr_c_seed.append(r2(y_tr,y1_c[:n_tr,:]))
    r2_val_c_seed.append(r2(y_val,y1_c[n_tr:,:]))
  r2_tr.append(r2_tr_seed)
  r2_val.append(r2_val_seed)
  r2_tr_c.append(r2_tr_c_seed)
  r2_val_c.append(r2_val_c_seed)
  axs.cla()
  axs.plot(1/np.array(sigma_ms[:len(r2_tr)]),np.nanmean(np.array(r2_tr),1,where=(np.array(r2_tr)>-100)),'C0')
  #axs[0].plot(1/np.array(sigma_ms[:len(r2_tr)]),np.quantile(np.array(r2_tr),0.8,1),'C0--')
  #axs[0].plot(1/np.array(sigma_ms[:len(r2_tr)]),np.quantile(np.array(r2_tr),0.2,1),'C0--')
  axs.plot(1/np.array(sigma_ms[:len(r2_val)]),np.nanmean(np.array(r2_val),1,where=(np.array(r2_val)>-100)),'C1')
  #axs[0].plot(1/np.array(sigma_ms[:len(r2_val)]),np.quantile(np.array(r2_val),0.8,1),'C1--')
  #axs[0].plot(1/np.array(sigma_ms[:len(r2_val)]),np.quantile(np.array(r2_val),0.2,1),'C1--')
  
  #axs[1].cla()
  axs.plot(1/np.array(sigma_ms[:len(r2_tr_c)]),np.nanmean(np.array(r2_tr_c),1,where=(np.array(r2_tr_c)>-100)),'C2--')
  #axs[1].plot(1/np.array(sigma_ms[:len(r2_tr_c)]),np.quantile(np.array(r2_tr_c),0.8,1),'C0--')
  #axs[1].plot(1/np.array(sigma_ms[:len(r2_tr_c)]),np.quantile(np.array(r2_tr_c),0.2,1),'C0--')
  axs.plot(1/np.array(sigma_ms[:len(r2_val_c)]),np.nanmean(np.array(r2_val_c),1,where=(np.array(r2_val_c)>-100)),'C3--')
  #axs[1].plot(1/np.array(sigma_ms[:len(r2_val_c)]),np.quantile(np.array(r2_val_c),0.8,1),'C1--')
  #axs[1].plot(1/np.array(sigma_ms[:len(r2_val_c)]),np.quantile(np.array(r2_val_c),0.2,1),'C1--')
  axs.set_ylim([-0.01,1.01])
  fig.savefig('figures/double_descent2_'+str(std)+'.pdf')
