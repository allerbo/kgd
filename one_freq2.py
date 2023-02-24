import numpy as np
from kgd import kgd
from matplotlib import pyplot as plt
import sys
from scipy.stats import wilcoxon
from scipy.optimize import minimize

def krr(xs,x_tr,y_tr_in,lbda,sigma):
  y_tr_mean=np.mean(y_tr_in)
  y_tr=y_tr_in-y_tr_mean
  Ks=np.exp(-0.5*np.square((xs-x_tr.T)/sigma))
  K=np.exp(-0.5*np.square((x_tr-x_tr.T)/sigma))
  return Ks@np.linalg.solve(K+lbda*np.eye(K.shape[0]),y_tr)+y_tr_mean

def r2(y,y_hat):
  return 1-np.mean((y-y_hat)**2)/np.mean((y-np.mean(y))**2)


def log_marg(x,y_in,lbda_seed, sigma_seed, lbda_bounds,sigma_bounds):
  y=y_in-np.mean(y_in)
  n=x.shape[0]
  def log_marg_fn(args):
    Kl=np.exp(-0.5*np.square((x-x.T)/args[1]))+args[0]*np.eye(n)
    return (y.T@np.linalg.solve(Kl,y) + np.log(np.linalg.det(Kl)))[0][0]
  
  res = minimize(log_marg_fn, [lbda_seed, sigma_seed], bounds=[lbda_bounds,sigma_bounds])
  return res.x

NS=500

def make_data(seed):
  np.random.seed(seed)
  FREQ1=1
  OBS_FREQ=10
  OBS_FREQ=80
  def fy(x):
    return np.sin(8*2*np.pi*x**2)
    #return np.sin(FREQ1*2*np.pi*x)
  x_tr=np.random.uniform(-1,1,2*FREQ1*OBS_FREQ).reshape((-1,1))
  y_tr=fy(x_tr)+np.random.normal(0,.1,x_tr.shape)
  #y_tr=fy(x_tr)+np.random.laplace(0,.2,x_tr.shape)
  #y_tr=fy(x_tr)+0.0*np.random.standard_cauchy(x_tr.shape)
  x_val=np.random.uniform(np.min(x_tr),np.max(x_tr),NS).reshape((-1,1))
  y_val=fy(x_val)
  return x_tr, y_tr, x_val, y_val

r2s_krr=[]
r2s_kgd=[]
for seed in range(100):
  x_tr, y_tr, x_val, y_val=make_data(seed)
  n_tr=x_tr.shape[0]
  xs=np.vstack((x_tr,x_val))
  xs_argsort=xs.argsort(0)
  
  lbda, sigma=log_marg(x_tr,y_tr,lbda_seed=0.001, sigma_seed=.01, lbda_bounds=[1e-6,10],sigma_bounds=[1e-6,10])
  y1_krr=krr(xs,x_tr,y_tr,lbda,sigma)
  r2s_krr.append(r2(y_val,y1_krr[n_tr:,:]))
  
  y1_kgd=kgd(xs,x_tr,y_tr, plot=False, step_size=0.01, sleep_time=0.1,val_data=[x_val, y_val])
  r2s_kgd.append(r2(y_val,y1_kgd[n_tr:,:]))
  print(lbda,sigma,r2s_krr[-1],r2s_kgd[-1])

  fig,ax=plt.subplots(1,1,figsize=(20,6))
  ax.plot(x_tr,y_tr,'ok')
  ax.plot(x_val,y_val,'or')
  ax.plot(xs[xs_argsort,0],y1_krr[xs_argsort,0],'C0')
  ax.plot(xs[xs_argsort,0],y1_kgd[xs_argsort,0],'C2')
  fig.savefig('figures/krr1.pdf')
  diff=np.array(r2s_kgd)-np.array(r2s_krr)
  print(f'KRR:  {np.quantile(r2s_krr,0.1):.2f}, {np.median(r2s_krr):.2f}, {np.quantile(r2s_krr,0.9):.2f}.')
  print(f'KGD:  {np.quantile(r2s_kgd,0.1):.2f}, {np.median(r2s_kgd):.2f}, {np.quantile(r2s_kgd,0.9):.2f}.')
  print(f'Diff: {np.quantile(diff,0.1):.2f}, {np.median(diff):.2f}, {np.quantile(diff,0.9):.2f}.')
  print(wilcoxon(r2s_kgd, r2s_krr, alternative='greater')[1])
  print('')
