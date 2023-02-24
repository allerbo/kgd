import numpy as np
from kgd import kgd
from matplotlib import pyplot as plt
from kgd import kgd
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from help_fcts import r2, krr, kern, gcv, log_marg, log_marg_ns
import sys

NS=500

def make_data(seed):
  np.random.seed(seed)
  N_TR=50
  def fy(x):
    return np.sin(5*2*np.pi*x*np.exp(-0.5*x))
  x_tr=np.random.exponential(.5,N_TR).reshape((-1,1))
  y_tr=fy(x_tr)+np.random.normal(0,.1,x_tr.shape)
  x_val=np.random.uniform(np.min(x_tr),np.max(x_tr),NS).reshape((-1,1))
  y_val=fy(x_val)
  return x_tr, y_tr, x_val, y_val


r2s_kgd=[]
r2s_krr_cv=[]
r2s_krr_lm=[]
r2s_krr_lm_ns=[]
for seed in range(100):
  print(seed)
  x_tr, y_tr, x_val, y_val=make_data(seed)
  n_tr=x_tr.shape[0]
  xs=np.vstack((x_tr,x_val))
  xs_argsort=xs.argsort(0)
  
  y1_kgd=kgd(xs,x_tr,y_tr, plot=False, step_size=0.01, sleep_time=0.1,val_data=[x_val, y_val])
  r2s_kgd.append(r2(y_val,y1_kgd[n_tr:,:]))
  
  lbda_cv, sigma_cv=gcv(x_tr,y_tr, np.geomspace(1e-4,.1,30),np.geomspace(1e-4,2,30))
  lbda_lm, sigma_lm=log_marg(x_tr,y_tr, lbda_bounds=[1e-4,.1],sigma_bounds=[1e-4,2], lbda_seed=lbda_cv, sigma_seed=sigma_cv)
  lbda_lm_ns, sigma_lm_ns=log_marg_ns(x_tr,y_tr, lbda_bounds=[1e-4,.1],sigma_bounds=[1e-4,2])
  y1_krr_cv=krr(xs,x_tr,y_tr,lbda_cv,sigma_cv)
  y1_krr_lm=krr(xs,x_tr,y_tr,lbda_lm,sigma_lm)
  y1_krr_lm_ns=krr(xs,x_tr,y_tr,lbda_lm_ns,sigma_lm_ns)
  r2s_krr_cv.append(r2(y_val,y1_krr_cv[n_tr:,:]))
  r2s_krr_lm.append(r2(y_val,y1_krr_lm[n_tr:,:]))
  r2s_krr_lm_ns.append(r2(y_val,y1_krr_lm_ns[n_tr:,:]))
  p_val_cv=wilcoxon(r2s_kgd, r2s_krr_cv, alternative='greater')[1]
  p_val_lm=wilcoxon(r2s_kgd, r2s_krr_lm, alternative='greater')[1]
  p_val_lm_ns=wilcoxon(r2s_kgd, r2s_krr_lm_ns, alternative='greater')[1]

  print(f'KGD:      {np.quantile(r2s_kgd,0.1):.2f}, {np.median(r2s_kgd):.2f}, {np.quantile(r2s_kgd,0.9):.2f}.')
  print(f'KRR CV:   {np.quantile(r2s_krr_cv,0.1):.2f}, {np.median(r2s_krr_cv):.2f}, {np.quantile(r2s_krr_cv,0.9):.2f}. Lambda: {lbda_cv:.3g}, Sigma: {sigma_cv:.3g}.')
  print(f'KRR LM:   {np.quantile(r2s_krr_lm,0.1):.2f}, {np.median(r2s_krr_lm):.2f}, {np.quantile(r2s_krr_lm,0.9):.2f}. Lambda: {lbda_lm:.3g}, Sigma: {sigma_lm:.3g}.')
  print(f'KRR LMNS: {np.quantile(r2s_krr_lm_ns,0.1):.2f}, {np.median(r2s_krr_lm_ns):.2f}, {np.quantile(r2s_krr_lm_ns,0.9):.2f}. Lambda: {lbda_lm_ns:.3g}, Sigma: {sigma_lm_ns:.3g}.')
  print(f'p cv:     {p_val_cv:.2f}.')
  print(f'p lm:     {p_val_lm:.2f}.')
  print(f'p lmns:   {p_val_lm_ns:.2f}.')
  print('')

  #lbda, sigma=log_marg(x_tr,y_tr, lbda_bounds=[1e-4,.1],sigma_bounds=[1e-4,2])
  #y1_krr=krr(xs,x_tr,y_tr,lbda,sigma)
  #r2s_krr.append(r2(y_val,y1_krr[n_tr:,:]))
  #
  #print(lbda,sigma,r2s_krr[-1],r2s_kgd[-1])
  
  fig,ax=plt.subplots(1,1,figsize=(20,6))
  ax.plot(x_tr,y_tr,'ok')
  ax.plot(x_val,y_val,'or')
  ax.plot(xs[xs_argsort,0],y1_krr_cv[xs_argsort,0],'C0')
  ax.plot(xs[xs_argsort,0],y1_krr_lm[xs_argsort,0],'C1')
  ax.plot(xs[xs_argsort,0],y1_krr_lm_ns[xs_argsort,0],'C3:')
  ax.plot(xs[xs_argsort,0],y1_kgd[xs_argsort,0],'C2')
  fig.savefig('figures/krr1.pdf')

