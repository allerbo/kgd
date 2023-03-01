import numpy as np
import pandas as pd
from kgd import kgd
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from help_fcts import r2, krr, kern, gcv, log_marg, log_marg_ns



#np.random.seed(0)
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
#for day in [21]:
LBDA_MIN=1e-4
LBDA_MAX=10
SIGMA_MIN=1e-2
SIGMA_MAX=10
SWEEP=False
if SWEEP:
  days=np.unique(data[:,1])
  #np.random.shuffle(days)
else:
  days=[1]
for day in days:
  data1=data[data[:,1]==day]
  np.random.shuffle(data1)
  X=data1[:,8:10]
  X=(X-np.mean(X, 0))/np.std(X,0)
  #y=data1[:,5].reshape((-1,1))
  y=data1[:,7].reshape((-1,1))
  n=X.shape[0]
  per=np.random.permutation(n)
  folds=np.array_split(per,5)
   
  r2s_kgd=[]
  r2s_krr_cv=[]
  r2s_krr_lm=[]
  r2s_krr_lm_ns=[]
  #for seed in range(10):
  for v_fold in range(len(folds)):
    t_idxs=np.concatenate([folds[t_fold] for t_fold in range(len(folds)) if v_fold != t_fold])
    v_idxs=folds[v_fold]

    X_tr=X[t_idxs,:]
    X_val=X[v_idxs,:]
    y_tr=y[t_idxs,:]
    y_val=y[v_idxs,:]
    n_tr=X_tr.shape[0]
    
    y1_kgd=kgd(np.vstack((X_tr,X_val)),X_tr,y_tr, plot=False, step_size=0.01)
    r2s_kgd.append(r2(y_val,y1_kgd[n_tr:,:]))
    
    lbda_cv, sigma_cv=gcv(X_tr,y_tr, np.geomspace(LBDA_MIN, LBDA_MAX,30),np.geomspace(SIGMA_MIN, SIGMA_MAX, 30))
    if lbda_cv in [LBDA_MIN, LBDA_MAX] or sigma_cv in [SIGMA_MIN, SIGMA_MAX]:
      print('OOOBS!',day, lbda_cv, sigma_cv)
    #lbda_cv, sigma_cv=gcv(X_tr,y_tr, np.geomspace(1e-3,1,30),np.geomspace(1,100,30))
    #lbda_lm, sigma_lm=log_marg(X_tr,y_tr, lbda_bounds=[0.1*LBDA_MIN, 10*LBDA_MAX],sigma_bounds=[0.1*SIGMA_MIN, 10*SIGMA_MAX], lbda_seed=lbda_cv, sigma_seed=sigma_cv)
    #lbda_lm_ns, sigma_lm_ns=log_marg_ns(X_tr,y_tr, lbda_bounds=[0.1*LBDA_MIN, 10*LBDA_MAX],sigma_bounds=[0.1*SIGMA_MIN, 10*SIGMA_MAX])
    y1_krr_cv=krr(np.vstack((X_tr, X_val)),X_tr,y_tr,lbda_cv,sigma_cv)
    #y1_krr_lm=krr(np.vstack((X_tr, X_val)),X_tr,y_tr,lbda_lm,sigma_lm)
    #y1_krr_lm_ns=krr(np.vstack((X_tr, X_val)),X_tr,y_tr,lbda_lm_ns,sigma_lm_ns)
    r2s_krr_cv.append(r2(y_val,y1_krr_cv[n_tr:,:]))
    #r2s_krr_lm.append(r2(y_val,y1_krr_lm[n_tr:,:]))
    #r2s_krr_lm_ns.append(r2(y_val,y1_krr_lm_ns[n_tr:,:]))
    p_val_cv=wilcoxon(r2s_kgd, r2s_krr_cv, alternative='greater')[1]
    #p_val_lm=wilcoxon(r2s_kgd, r2s_krr_lm, alternative='greater')[1]
    #p_val_lm_ns=wilcoxon(r2s_kgd, r2s_krr_lm_ns, alternative='greater')[1]
    p_val_vc=wilcoxon(r2s_krr_cv, r2s_kgd, alternative='greater')[1]
    
    if not SWEEP:
      print(f'KGD:      {np.quantile(r2s_kgd,0.1):.2f}, {np.median(r2s_kgd):.2f}, {np.quantile(r2s_kgd,0.9):.2f}.')
      print(f'KRR CV:   {np.quantile(r2s_krr_cv,0.1):.2f}, {np.median(r2s_krr_cv):.2f}, {np.quantile(r2s_krr_cv,0.9):.2f}. Lambda: {lbda_cv:.3g}, Sigma: {sigma_cv:.3g}.')
      #print(f'KRR LM:   {np.quantile(r2s_krr_lm,0.1):.2f}, {np.median(r2s_krr_lm):.2f}, {np.quantile(r2s_krr_lm,0.9):.2f}. Lambda: {lbda_lm:.3g}, Sigma: {sigma_lm:.3g}.')
      #print(f'KRR LMNS: {np.quantile(r2s_krr_lm_ns,0.1):.2f}, {np.median(r2s_krr_lm_ns):.2f}, {np.quantile(r2s_krr_lm_ns,0.9):.2f}. Lambda: {lbda_lm_ns:.3g}, Sigma: {sigma_lm_ns:.3g}.')
      print(f'p cv:     {p_val_cv:.2f}.')
      #print(f'p lm:     {p_val_lm:.2f}.')
      #print(f'p lmns:   {p_val_lm_ns:.2f}.')
      #print(np.mean(y), np.std(y))
      print('')
  if SWEEP:
    #print(day, p_val_cv, p_val_lm, p_val_lm_ns, np.mean(y), np.std(y))
    print(day, np.round(p_val_cv,3), np.round(p_val_vc,3))
  
