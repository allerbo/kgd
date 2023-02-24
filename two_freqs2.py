import numpy as np
from kgd import kgd
from matplotlib import pyplot as plt
import sys
from scipy.stats import wilcoxon

def krr(xs,x_tr,y_tr_in,lbda,sigma):
  y_tr_mean=np.mean(y_tr_in)
  y_tr=y_tr_in-y_tr_mean
  Ks=np.exp(-0.5*np.square((xs-x_tr.T)/sigma))
  K=np.exp(-0.5*np.square((x_tr-x_tr.T)/sigma))
  return Ks@np.linalg.solve(K+lbda*np.eye(K.shape[0]),y_tr)+y_tr_mean

def r2(y,y_hat):
  return 1-np.mean((y-y_hat)**2)/np.mean((y-np.mean(y))**2)


NS=500

def make_data(seed):
  np.random.seed(seed)
  FREQ1=1
  FREQ2=10
  OBS_FREQ=20
  def fy(x):
    return np.sin(FREQ1*2*np.pi*x)*(np.abs(x)>0.5)+np.sin(FREQ2*2*np.pi*x)*(np.abs(x)<0.5)
  x_tr1=np.random.uniform(-2,2,2*FREQ1*OBS_FREQ).reshape((-1,1))
  x_tr2=np.random.uniform(-.5,.5,FREQ2*OBS_FREQ//2).reshape((-1,1))
  x_tr=np.vstack((x_tr1,x_tr2))
  y_tr=fy(x_tr)+np.random.normal(0,.1,x_tr.shape)
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
  
  r2_max=0
  for lbda in np.geomspace(1,0.001, 50):
    for sigma in np.geomspace(.1,0.01,50):
      y1_krr=krr(xs,x_tr,y_tr,lbda,sigma)
      r2_krr=r2(y_val,y1_krr[n_tr:,:])
      if r2_krr>r2_max:
        r2_max=r2_krr
        best_vals=[lbda,sigma,r2(y_tr, y1_krr[:n_tr,:]),r2_max]
  r2s_krr.append(r2_max)
  #print(best_vals)
  
  y1_kgd=kgd(xs,x_tr,y_tr, plot=False, step_size=0.01, sleep_time=0.1,val_data=[x_val, y_val])
  r2s_kgd.append(r2(y_val,y1_kgd[n_tr:,:]))
  #print(r2_kgd)

  y1_krr=krr(xs,x_tr,y_tr,best_vals[0], best_vals[1])
  fig,ax=plt.subplots(1,1,figsize=(20,6))
  ax.plot(x_tr,y_tr,'ok')
  ax.plot(x_val,y_val,'or')
  ax.plot(xs[xs_argsort,0],y1_krr[xs_argsort,0],'C0')
  ax.plot(xs[xs_argsort,0],y1_kgd[xs_argsort,0],'C2')
  fig.savefig('figures/krr.pdf')
  diff=np.array(r2s_kgd)-np.array(r2s_krr)
  print(f'KRR:  {np.quantile(r2s_krr,0.1):.2f}, {np.median(r2s_krr):.2f}, {np.quantile(r2s_krr,0.9):.2f}.')
  print(f'KGD:  {np.quantile(r2s_kgd,0.1):.2f}, {np.median(r2s_kgd):.2f}, {np.quantile(r2s_kgd,0.9):.2f}.')
  print(f'Diff: {np.quantile(diff,0.1):.2f}, {np.median(diff):.2f}, {np.quantile(diff,0.9):.2f}.')
  print(wilcoxon(r2s_kgd, r2s_krr, alternative='greater')[1])
  print('')
