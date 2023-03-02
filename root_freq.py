import numpy as np
from kgd import kgd
from matplotlib import pyplot as plt
from kgd import kgd
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from help_fcts import r2, krr, kern, gcv, log_marg, log_marg_ns
import sys

NS=500

#def make_data(seed):
#  np.random.seed(seed)
#  N_TR=100
#  def fy(x):
#    #return np.sin(2*np.pi*np.sqrt((x+1)))
#    #return np.sin(2*np.pi*(x+1)**0.3)
#    return np.sin(2*np.pi*np.sign(x)*np.abs(x)**(1/3))
#  #x_tr=np.random.exponential(10,(N_TR,1))
#  x_tr=np.random.normal(0,50,(N_TR,1))
#  y_tr=fy(x_tr)+np.random.normal(0,.2,x_tr.shape)
#  x_val=np.random.uniform(np.min(x_tr), np.max(x_tr), NS).reshape((-1,1))
#  y_val=fy(x_val)
#  #x1=np.linspace(np.min(x_tr), np.max(x_tr), 1000).reshape((-1,1))
#  x1=np.linspace(-5**3, 5**3, 1000).reshape((-1,1))
#  y1=fy(x1)
#  lbda_bounds=[1e-4,.1]
#  sigma_bounds=[1e-2,100]
#  return x_tr, y_tr, x_val, y_val, x1, y1, lbda_bounds, sigma_bounds

def shift_root(x,power,shift):
  return np.sign(x)*((np.abs(x)+shift)**power-shift**power)
  
def make_data(seed=None):
  if not seed is None:
    np.random.seed(seed)
  N_TR=100
  def fy(x):
    #return np.sin(4*2*np.pi*np.sign(x)*np.abs(x)**(1/4))
    return np.sin(4*2*np.pi*shift_root(x,1/4,0))
    #return np.sin(10*2*np.pi*shift_root(x,1/10,0.1))
  x_tr=np.random.normal(0,1,(N_TR,1))
  y_tr=fy(x_tr)+np.random.normal(0,.2,x_tr.shape)
  x_val=np.random.uniform(np.min(x_tr), np.max(x_tr), NS).reshape((-1,1))
  #x_val=np.linspace(np.min(x_tr), np.max(x_tr), NS).reshape((-1,1))
  y_val=fy(x_val)
  x1=np.linspace(-3, 3, 1000).reshape((-1,1))
  y1=fy(x1)
  lbda_bounds=[1e-3,1]
  sigma_bounds=[1e-2,1]
  return x_tr, y_tr, x_val, y_val, x1, y1, lbda_bounds, sigma_bounds

#x_tr=sample(1000)
#fig,ax=plt.subplots(1,1,figsize=(20,6))
#ax.hist(x_tr,50)
#fig.savefig('figures/sqrt_freq.pdf')
#sys.exit()

r2s_kgd=[]
r2s_krr_cv=[]
r2s_krr_lm=[]
r2s_krr_lm_ns=[]
for seed in range(100):
  print(seed)
  #x_tr, y_tr, x_val, y_val, x1,y1, lbda_bounds, sigma_bounds=make_data(seed)
  x_tr, y_tr, x_val, y_val, x1,y1, lbda_bounds, sigma_bounds=make_data()
  n_tr=x_tr.shape[0]
  xs=np.vstack((x_tr,x_val))
  xs_argsort=xs.argsort(0)
  
  y1_kgd=kgd(xs,x_tr,y_tr, plot=False, step_size=0.01, sleep_time=0.1,val_data=[x_val, y_val])
  r2s_kgd.append(r2(y_val,y1_kgd[n_tr:,:]))
  
  lbda_cv, sigma_cv=gcv(x_tr,y_tr, lbda_bounds, sigma_bounds)
  y1_krr_cv=krr(xs,x_tr,y_tr,lbda_cv,sigma_cv)
  r2s_krr_cv.append(r2(y_val,y1_krr_cv[n_tr:,:]))
  p_val_cv=wilcoxon(r2s_kgd, r2s_krr_cv, alternative='greater')[1]
  
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
  print(f'KRR LM:   {np.quantile(r2s_krr_lm,0.1):.2f}, {np.median(r2s_krr_lm):.2f}, {np.quantile(r2s_krr_lm,0.9):.2f}. Lambda: {lbda_lm:.3g}, Sigma: {sigma_lm:.3g}.')
  print(f'KRR LMNS: {np.quantile(r2s_krr_lm_ns,0.1):.2f}, {np.median(r2s_krr_lm_ns):.2f}, {np.quantile(r2s_krr_lm_ns,0.9):.2f}. Lambda: {lbda_lm_ns:.3g}, Sigma: {sigma_lm_ns:.3g}.')
  print(f'p cv:     {p_val_cv:.2f}.')
  print(f'p lm:     {p_val_lm:.2f}.')
  print(f'p lmns:   {p_val_lm_ns:.2f}.')
  print('')

  
  fig,ax=plt.subplots(1,1,figsize=(20,6))
  ax.plot(x1,y1,'C7',lw=3)
  ax.plot(x_tr,y_tr,'ok')
  #ax.plot(x_val,y_val,'or')
  ax.plot(xs[xs_argsort,0],y1_krr_cv[xs_argsort,0],'C0')
  ax.plot(xs[xs_argsort,0],y1_krr_lm[xs_argsort,0],'C1')
  ax.plot(xs[xs_argsort,0],y1_krr_lm_ns[xs_argsort,0],'C3:')
  ax.plot(xs[xs_argsort,0],y1_kgd[xs_argsort,0],'C2')
  fig.savefig('figures/sqrt_freq.pdf')
  plt.close()

