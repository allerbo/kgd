import numpy as np
from kgd import kgd
from matplotlib import pyplot as plt
from kgd import kgd
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from help_fcts import r2, krr, kern, gcv, log_marg, log_marg_ns
import sys
from time import sleep

NS=500


def make_data3(freq, n, seed=None):
  if not seed is None:
    np.random.seed(seed)
  def fy(x):
    return x**2
  x_tr=np.random.uniform(-3,3,n).reshape((-1,1))
  y_tr=fy(x_tr)+np.random.normal(0,2,x_tr.shape)
  x_val=np.linspace(-3,3,500).reshape((-1,1))
  x1=np.linspace(-3, 3, 1000).reshape((-1,1))
  y1=fy(x1)
  return x_tr, y_tr, x_val, x1, y1

def make_data(freq, n, seed=None):
  if not seed is None:
    np.random.seed(seed)
  def fy(x):
    return np.sin(freq*2*np.pi*x)
  x_tr=np.random.uniform(-1,1,n).reshape((-1,1))
  y_tr=fy(x_tr)+np.random.normal(0,.2,x_tr.shape)
  x_val=np.linspace(-1,1,500).reshape((-1,1))
  x1=np.linspace(-1, 1, 1000).reshape((-1,1))
  y1=fy(x1)
  return x_tr, y_tr, x_val, x1, y1

def make_data2(a, b, seed=None):
  if not seed is None:
    np.random.seed(seed)
  FREQ1=1
  FREQ2=5
  OBS_FREQ=30
  def fy(x):
    return np.sin(FREQ1*2*np.pi*x)*(x<0)+np.sin(FREQ2*2*np.pi*x)*(x>0)
  x_tr1=np.random.uniform(-2,0,FREQ1*OBS_FREQ).reshape((-1,1))
  y_tr1=fy(x_tr1)+np.random.normal(0,.2,x_tr1.shape)
  x_tr2=np.random.uniform(0,2,FREQ2*OBS_FREQ).reshape((-1,1))
  y_tr2=fy(x_tr2)+np.random.normal(0,.2,x_tr2.shape)
  x_tr=np.vstack((x_tr1,x_tr2))
  y_tr=np.vstack((y_tr1,y_tr2))
  x_val=np.random.uniform(-2,2,500).reshape((-1,1))
  x1=np.linspace(-2, 2, 1000).reshape((-1,1))
  y1=fy(x1)
  return x_tr, y_tr, x_val, x1, y1

def make_data1(a,b,seed=None):
  if not seed is None:
    np.random.seed(seed)
  FREQ1=1
  FREQ2=10
  N_TR=20
  def fy(x):
    return np.sin(FREQ1*2*np.pi*x)*(x<0)+np.sin(FREQ2*2*np.pi*x)*(x>0)
  x_tr1=np.random.uniform(-2,0,N_TR).reshape((-1,1))
  y_tr1=fy(x_tr1)+np.random.normal(0,.2,x_tr1.shape)
  x_tr2=np.random.uniform(0,.2,N_TR).reshape((-1,1))
  y_tr2=fy(x_tr2)+np.random.normal(0,.2,x_tr2.shape)
  x_tr=np.vstack((x_tr1,x_tr2))
  y_tr=np.vstack((y_tr1,y_tr2))
  x_val=np.random.uniform(-2,.2,NS).reshape((-1,1))
  x1=np.linspace(-2, .2, 1000).reshape((-1,1))
  y1=fy(x1)
  return x_tr, y_tr, x_val, x1, y1


FREQ=7
N=100
seed=1
x_tr, y_tr, x_val, x1, y1=make_data3(FREQ, N, seed)
xs=np.vstack((x_tr,x_val))
xs_argsort=xs.argsort(0)

y1_kgd, sigmas, r2s =kgd(xs,x_tr,y_tr, path=True, plot=True)

sigma_half=sigmas[np.argmin((np.array(r2s)-0.5)**2)]

fig,axs=plt.subplots(2,1,figsize=(20,12))
axs[0].plot(x1,y1,'C7',lw=3)
axs[0].plot(x_tr,y_tr,'ok')
axs[0].plot(xs[xs_argsort,0],y1_kgd[xs_argsort,0],'C2')
axs[0].plot([np.min(x_tr),np.min(x_tr)+sigma_half],2*[1.1*np.min(y1_kgd)],'C3',lw=3)
axs[1].plot(sigmas,r2s)
#axs[2].plot(r2s,sigmas)
axs[1].set_xscale('log')
fig.savefig('figures/find_freqs.pdf')


