import numpy as np
import pandas as pd
from kgd import kgd
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from help_fcts import r2, krr, kern, gcv, log_marg, log_marg_ns



data=pd.read_csv('bs_2000.csv',sep=',').to_numpy()
#max_obs=0
#for d in np.unique(data[:,1]):
#  n_obs=data[data[:,1]==d].shape[0]
#  if n_obs>max_obs:
#    max_obs=n_obs
#    days=[d]
#  elif n_obs==max_obs:
#    days.append(d)
#
##86
##21
#data=data[data[:,1]==1]
#data=data[data[:,1]==2]
#data=data[data[:,1]==3]
#for day in np.unique(data[:,1]):
for day in [21]:
  data1=data[data[:,1]==day]
  
  r2s_kgd=[]
  r2s_krr_cv=[]
  r2s_krr_lm=[]
  r2s_krr_lm_ns=[]
  for seed in range(10):
    np.random.seed(seed)
    np.random.shuffle(data1)
    
    X=data1[:,8:10]
    y=data1[:,7].reshape((-1,1))
    
    n_tot=data1.shape[0]
    n_tr=round(0.7*n_tot)
    
    X_tr=X[:n_tr,:]
    X_val=X[n_tr:,:]
    y_tr=y[:n_tr,:]
    y_val=y[n_tr:,:]
    
    y1_kgd=kgd(X,X_tr,y_tr, plot=False, step_size=0.01)
    r2s_kgd.append(r2(y_val,y1_kgd[n_tr:,:]))
    
    #lbda, sigma=log_marg_ns(X_tr,y_tr, lbda_bounds=[1,1e5],sigma_bounds=[1,1e3])
  
    lbda_cv, sigma_cv=gcv(X_tr,y_tr, np.geomspace(1e-3,1,30),np.geomspace(1,100,30))
    lbda_lm, sigma_lm=log_marg(X_tr,y_tr, lbda_bounds=[1e-3,10],sigma_bounds=[1,100], lbda_seed=lbda_cv, sigma_seed=sigma_cv)
    lbda_lm_ns, sigma_lm_ns=log_marg_ns(X_tr,y_tr, lbda_bounds=[1e-3,10],sigma_bounds=[1,100])
    y1_krr_cv=krr(X,X_tr,y_tr,lbda_cv,sigma_cv)
    y1_krr_lm=krr(X,X_tr,y_tr,lbda_lm,sigma_lm)
    y1_krr_lm_ns=krr(X,X_tr,y_tr,lbda_lm_ns,sigma_lm_ns)
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
  print(day, p_val_cv)
  
