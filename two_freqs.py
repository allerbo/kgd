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

def make_data1(seed=None):
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
  y_val=fy(x_val)
  x1=np.linspace(-2, .2, 1000).reshape((-1,1))
  y1=fy(x1)
  lbda_bounds=[0.001,1]
  sigma_bounds=[0.01,0.1]
  return x_tr, y_tr, x_val, y_val, x1, y1, lbda_bounds, sigma_bounds

def make_data2(seed=None):
  if not seed is None:
    np.random.seed(seed)
  FREQ1=1
  FREQ2=10
  OBS_FREQ=20
  def fy(x):
    return np.sin(FREQ1*2*np.pi*x)*(x<0)+np.sin(FREQ2*2*np.pi*x)*(x>0)
  x_tr1=np.random.uniform(-2,0,FREQ1*OBS_FREQ).reshape((-1,1))
  y_tr1=fy(x_tr1)+np.random.normal(0,.2,x_tr1.shape)
  x_tr2=np.random.uniform(0,2,FREQ2*OBS_FREQ).reshape((-1,1))
  y_tr2=fy(x_tr2)+np.random.normal(0,.2,x_tr2.shape)
  x_tr=np.vstack((x_tr1,x_tr2))
  y_tr=np.vstack((y_tr1,y_tr2))
  x_val=np.random.uniform(-2,2,NS).reshape((-1,1))
  y_val=fy(x_val)
  x1=np.linspace(-2, 2, 1000).reshape((-1,1))
  y1=fy(x1)
  lbda_bounds=[1e-6,0.1]
  sigma_bounds=[0.01,0.1]
  return x_tr, y_tr, x_val, y_val, x1, y1, lbda_bounds, sigma_bounds


LM=True

fig,axs=plt.subplots(2,1,figsize=(20,12))
r2s_kgd=[]
r2s_krr_cv=[]
r2s_krr_lm=[]
r2s_krr_lm_ns=[]
for seed in range(100):
  print('\n',seed)
  for ii, (make_data, ax) in enumerate(zip([make_data2, make_data1], axs)):
    x_tr, y_tr, x_val, y_val, x1,y1, lbda_bounds, sigma_bounds=make_data(seed)
    #x_tr, y_tr, x_val, y_val, x1,y1, lbda_bounds, sigma_bounds=make_data()
    n_tr=x_tr.shape[0]
    xs=np.vstack((x_tr,x_val))
    xs_argsort=xs.argsort(0)
    
    y1_kgd=kgd(xs,x_tr,y_tr, plot=False, step_size=0.01, sleep_time=0.1,val_data=[x_val, y_val])
    r2s_kgd.append(r2(y_val,y1_kgd[n_tr:,:]))
    
    lbda_cv, sigma_cv=gcv(x_tr,y_tr, lbda_bounds, sigma_bounds, n_lbdas=30, n_sigmas=30)
    y1_krr_cv=krr(xs,x_tr,y_tr,lbda_cv,sigma_cv)
    r2s_krr_cv.append(r2(y_val,y1_krr_cv[n_tr:,:]))
    p_val_cv=wilcoxon(r2s_kgd, r2s_krr_cv, alternative='greater')[1]
    
    if LM:
      lbda_lm, sigma_lm=log_marg(x_tr,y_tr, [0.1*lbda_bounds[0], 10*lbda_bounds[1]],[0.1*sigma_bounds[0], 10*sigma_bounds[1]], lbda_seed=lbda_cv, sigma_seed=sigma_cv)
      y1_krr_lm=krr(xs,x_tr,y_tr,lbda_lm,sigma_lm)
      r2s_krr_lm.append(r2(y_val,y1_krr_lm[n_tr:,:]))
      p_val_lm=wilcoxon(r2s_kgd, r2s_krr_lm, alternative='greater')[1]
       
      lbda_lm_ns, sigma_lm_ns=log_marg_ns(x_tr,y_tr, [0.1*lbda_bounds[0], 10*lbda_bounds[1]],[0.1*sigma_bounds[0], 10*sigma_bounds[1]])
      y1_krr_lm_ns=krr(xs,x_tr,y_tr,lbda_lm_ns,sigma_lm_ns)
      r2s_krr_lm_ns.append(r2(y_val,y1_krr_lm_ns[n_tr:,:]))
      p_val_lm_ns=wilcoxon(r2s_kgd, r2s_krr_lm_ns, alternative='greater')[1]
  
    print(f'KGD:      {np.quantile(r2s_kgd,0.1):.2f}, {np.median(r2s_kgd):.2f}, {np.quantile(r2s_kgd,0.9):.2f}.')
    print(f'KRR CV:   {np.quantile(r2s_krr_cv,0.1):.2f}, {np.median(r2s_krr_cv):.2f}, {np.quantile(r2s_krr_cv,0.9):.2f}. Lambda: {lbda_cv:.3g}, Sigma: {sigma_cv:.3g}.')
    print(f'p cv:     {p_val_cv:.2f}.')
    if LM:
      print(f'KRR LM:   {np.quantile(r2s_krr_lm,0.1):.2f}, {np.median(r2s_krr_lm):.2f}, {np.quantile(r2s_krr_lm,0.9):.2f}. Lambda: {lbda_lm:.3g}, Sigma: {sigma_lm:.3g}.')
      print(f'p lm:     {p_val_lm:.2f}.')
      print(f'KRR LMNS: {np.quantile(r2s_krr_lm_ns,0.1):.2f}, {np.median(r2s_krr_lm_ns):.2f}, {np.quantile(r2s_krr_lm_ns,0.9):.2f}. Lambda: {lbda_lm_ns:.3g}, Sigma: {sigma_lm_ns:.3g}.')
      print(f'p lmns:   {p_val_lm_ns:.2f}.')
      print('')
    
    ax.cla()
    ax.plot(x1,y1,'C7',lw=3)
    ax.plot(x_tr,y_tr,'ok')
    #ax.plot(x_val,y_val,'or')
    ax.plot(xs[xs_argsort,0],y1_krr_cv[xs_argsort,0],'C0')
    ax.plot(xs[xs_argsort,0],y1_kgd[xs_argsort,0],'C2')
    if LM:
      ax.plot(xs[xs_argsort,0],y1_krr_lm[xs_argsort,0],'C1:')
      ax.plot(xs[xs_argsort,0],y1_krr_lm_ns[xs_argsort,0],'C3:')
    ax.set_xlim([-2.1,2.1])
    ax.set_ylim([-1.7,1.7])
    fig.savefig('figures/two_freqs.pdf')

