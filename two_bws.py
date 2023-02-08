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
#x1=np.linspace(-2,2,1000).reshape((-1,1))

def f(x,ps):
  y=np.zeros(x.shape)
  y+=np.sin(ps[0]*2*np.pi*x)*((x>-2)&(x<0))
  y+=np.sin(ps[1]*2*np.pi*x)*((x>0)&(x<2))
  return y

#axs.plot(x1,f(x1,ps))
#axs.plot(x_tr,y_tr,'ok')
#axs.plot(x_val,y_val,'or')
#fig.savefig('figures/test.pdf')



L_FI=2
N_PER_FI=20

ps=[1,8]

step_size=0.1
sigma_0=4
sigma_m=1e-5
alpha=20
t=1e2
lbda=1e-4
gamma=1-t*lbda

#sigmas_krr=np.array([20.11,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02])
sigmas_krr=np.arange(1,0.01,-0.01)
sigmas_krr=np.geomspace(1,0.01,100)
sigmas_krr=np.geomspace(2,0.005,100)
r2_tr_krr={}
r2_val_krr={}
for sigma_krr in sigmas_krr:
  r2_tr_krr[sigma_krr]=[]
  r2_val_krr[sigma_krr]=[]

#axs1.cla()
#axs1.plot(x_tr,y_tr,'ok')
#axs1.plot(x_val,y_val,'xr')
#fig1.savefig('figures/kern_compl1_b.pdf')

n_val=200
r2_tr_seed=[]
r2_val_seed=[]
r2_tr_seed_krr={}
r2_val_seed_krr={}
r2_val_seed_diff={}
for sigma_krr in sigmas_krr:
  r2_tr_seed_krr[sigma_krr]=[]
  r2_val_seed_krr[sigma_krr]=[]
  r2_val_seed_diff[sigma_krr]=[]

for seed in range(100):
  np.random.seed(seed)
  x_tr=np.hstack((np.random.uniform(-2,0,N_PER_FI*ps[0]),np.random.uniform(0,2,N_PER_FI*ps[1]))).reshape((-1,1))
  n_tr=x_tr.shape[0]
  y_tr=f(x_tr,ps)+np.random.normal(0,.2,x_tr.shape)
  x_val=np.random.uniform(-2,2,n_val).reshape((-1,1))
  y_val=f(x_val,ps)#+np.random.normal(0,0,x_val.shape)
  
  for sigma_krr in sigmas_krr:
    y1_tr_krr=kern_gauss(x_tr,x_tr,sigma_krr)@np.linalg.solve(kern_gauss(x_tr,x_tr,sigma_krr)+lbda*np.eye(n_tr),y_tr)
    y1_val_krr=kern_gauss(x_val,x_tr,sigma_krr)@np.linalg.solve(kern_gauss(x_tr,x_tr,sigma_krr)+lbda*np.eye(n_tr),y_tr)
    r2_tr_seed_krr[sigma_krr].append(r2(y_tr,y1_tr_krr))
    r2_val_seed_krr[sigma_krr].append(r2(y_val,y1_val_krr))
   
  xs=np.vstack((x_tr,x_val))
  xs_argsort=xs.argsort(0)
  y1=np.zeros(xs.shape)
  sigma=sigma_0
  for i in range(int(t/step_size)):
    Ks=kern_gauss(xs,x_tr,sigma)
    gd_obj=gd_alg(Ks,y_tr,'gd','pred',step_size, gamma=gamma, var0=y1)
    if np.max(np.abs(y1))<1000:
      gd_obj.gd_step()
    y1=gd_obj.get_fs()
    #sigma=(sigma+sigma_m*alpha*step_size)/(1+alpha*step_size)
    sigma=(sigma-sigma_m**2*alpha*step_size)/(1+alpha*step_size*(sigma-2*sigma_m))
    #if i%100==0:
    #  print(i,int(t/step_size))
  #axs1.cla()
  #axs1.plot(x_tr,y_tr,'ok')
  #axs1.plot(x_val,y_val,'xr')
  #axs1.plot(xs[xs_argsort,0],y1[xs_argsort,0])
  #axs1.set_ylim([-2,2])
  #fig1.savefig('figures/kern_compl1_b.pdf')
  #sleep(.1)
  print(seed,alpha,r2(y_tr,y1[:n_tr,:]),r2(y_val,y1[n_tr:,:]))
  r2_tr_seed.append(r2(y_tr,y1[:n_tr,:]))
  r2_val_seed.append(r2(y_val,y1[n_tr:,:]))
  for sigma_krr in sigmas_krr:
    r2_val_seed_diff[sigma_krr].append(r2_val_seed[-1]-r2_val_seed_krr[sigma_krr][-1])

  axs1.cla()
  r2_val_median_diffs=[]
  r2_val_lq4_diffs=[]
  r2_val_uq4_diffs=[]
  r2_val_lq10_diffs=[]
  r2_val_uq10_diffs=[]
  for c,sigma_krr in enumerate(sigmas_krr):
    r2_val_median_diffs.append(np.median(np.array(r2_val_seed_diff[sigma_krr])))
    r2_val_lq4_diffs.append(np.quantile(np.array(r2_val_seed_diff[sigma_krr]),0.25))
    r2_val_uq4_diffs.append(np.quantile(np.array(r2_val_seed_diff[sigma_krr]),0.75))
    r2_val_lq10_diffs.append(np.quantile(np.array(r2_val_seed_diff[sigma_krr]),0.9))
    r2_val_uq10_diffs.append(np.quantile(np.array(r2_val_seed_diff[sigma_krr]),0.1))
  axs1.plot(sigmas_krr,r2_val_median_diffs,'C1')
  axs1.plot(sigmas_krr,r2_val_lq4_diffs,'C1--')
  axs1.plot(sigmas_krr,r2_val_uq4_diffs,'C1--')
  axs1.plot(sigmas_krr,r2_val_lq10_diffs,'C1:')
  axs1.plot(sigmas_krr,r2_val_uq10_diffs,'C1:')
  axs1.set_xscale('log')
  axs1.set_ylim([-1,1])
  axs1.axhline(0,color='k')
  fig1.savefig('figures/two_bws_diff.pdf')
  
  axs.cla()
  r2_tr_median=np.median(np.array(r2_tr_seed))
  r2_tr_lq=np.quantile(np.array(r2_tr_seed),0.25)
  r2_tr_uq=np.quantile(np.array(r2_tr_seed),0.75)
  r2_val_median=np.median(np.array(r2_val_seed))
  r2_val_lq=np.quantile(np.array(r2_val_seed),0.25)
  r2_val_uq=np.quantile(np.array(r2_val_seed),0.75)
  
  axs.plot(sigmas_krr,len(sigmas_krr)*[r2_tr_median],'C0')
  axs.plot(sigmas_krr,len(sigmas_krr)*[r2_tr_lq],'C0--')
  axs.plot(sigmas_krr,len(sigmas_krr)*[r2_tr_uq],'C0--')
  
  axs.plot(sigmas_krr,len(sigmas_krr)*[r2_val_median],'C2')
  axs.plot(sigmas_krr,len(sigmas_krr)*[r2_val_lq],'C2--')
  axs.plot(sigmas_krr,len(sigmas_krr)*[r2_val_uq],'C2--')
  
  r2_tr_median_krrs=[]
  r2_tr_lq_krrs=[]
  r2_tr_uq_krrs=[]
  r2_val_median_krrs=[]
  r2_val_lq_krrs=[]
  r2_val_uq_krrs=[]
  for c,sigma_krr in enumerate(sigmas_krr):
    r2_tr_median_krrs.append(np.median(np.array(r2_tr_seed_krr[sigma_krr])))
    r2_tr_lq_krrs.append(np.quantile(np.array(r2_tr_seed_krr[sigma_krr]),0.25))
    r2_tr_uq_krrs.append(np.quantile(np.array(r2_tr_seed_krr[sigma_krr]),0.75))
    r2_val_median_krrs.append(np.median(np.array(r2_val_seed_krr[sigma_krr])))
    r2_val_lq_krrs.append(np.quantile(np.array(r2_val_seed_krr[sigma_krr]),0.25))
    r2_val_uq_krrs.append(np.quantile(np.array(r2_val_seed_krr[sigma_krr]),0.75))
  
  axs.plot(sigmas_krr,r2_tr_median_krrs,'C1')
  axs.plot(sigmas_krr,r2_tr_lq_krrs,'C1--')
  axs.plot(sigmas_krr,r2_tr_uq_krrs,'C1--')

  axs.plot(sigmas_krr,r2_val_median_krrs,'C3')
  axs.plot(sigmas_krr,r2_val_lq_krrs,'C3--')
  axs.plot(sigmas_krr,r2_val_uq_krrs,'C3--')
  
  axs.set_ylim([-0.01,1.01])
  axs.set_xscale('log')
  
  fig.savefig('figures/two_bws.pdf')

