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

def f(x,n_per):
  return np.sin(n_per*2*np.pi*x)

fig_val,axs_val=plt.subplots(1,1,figsize=(12,6))
fig_tr,axs_tr=plt.subplots(1,1,figsize=(12,6))
fig_expl,axs_expl=plt.subplots(1,2,figsize=(14,4))


step_size=0.1

sigma_0=2
sigma_m=1e-5
alpha=20
t=1e2
lbda=1e-4
gamma=1-lbda*t


sigmas_krr=np.array([0.2,0.05])
sigma_cs=np.array(['C1','C3'])

labs = ['True Function', 'Observed Data','KGD']

lines=[Line2D([0],[0],color='C7',lw=3),plt.plot(0,0,'ok')[0],Line2D([0],[0],color='C0',lw=3)]
plt.cla()
for sigma_krr, sigma_c in zip(sigmas_krr,sigma_cs):
  lines.append(Line2D([0],[0],color=sigma_c,lw=3))
  labs.append('$\\sigma='+str(sigma_krr)+'$')


seed=1
seed=0
x1=np.linspace(-1,1,301).reshape((-1,1))
for ax, n_per in zip(axs_expl.ravel(),[1,8]):
  np.random.seed(seed)
  n_tr=20*n_per
  x_tr=np.random.uniform(-1,1,n_tr).reshape((-1,1))
  y_tr=f(x_tr,n_per)+np.random.normal(0,.2,x_tr.shape)
  y_true=f(x1,n_per)
  xs=np.vstack((x_tr,x1))
  xs_argsort=xs.argsort(0)
  ax.plot(x_tr,y_tr,'ok')
  ax.plot(x1, y_true, 'C7')
  
  sigma=sigma_0
#  y1=np.zeros(xs.shape)
#  for i in range(int(t/step_size)):
#    Ks=kern_gauss(xs,x_tr,sigma)
#    gd_obj=gd_alg(Ks,y_tr,'gd','pred',step_size, gamma=gamma,var0=y1)
#    if np.max(np.abs(y1))<1000:
#      gd_obj.gd_step()
#    y1=gd_obj.get_fs()
#    sigma=(sigma-sigma_m**2*alpha*step_size)/(1+alpha*step_size*(sigma-2*sigma_m))
#  ax.plot(xs[xs_argsort,0],y1[xs_argsort,0])
  for sigma_krr,sigma_c in zip(sigmas_krr,sigma_cs):
    y1_krr=kern_gauss(xs,x_tr,sigma_krr)@np.linalg.solve(kern_gauss(x_tr,x_tr,sigma_krr)+lbda*np.eye(n_tr),y_tr)
    ax.plot(xs[xs_argsort,0],y1_krr[xs_argsort,0], sigma_c)
  fig_expl.legend(lines, labs, loc='lower center', ncol=len(labs))
  fig_expl.tight_layout()
  fig_expl.subplots_adjust(bottom=.2)
  fig_expl.savefig('figures/sweep_bw_expl.pdf')

#sys.exit()


r2_tr=[]
r2_val=[]

sigmas_krr=np.array([1,0.3,0.1,0.03,0.01,0.003,0.001])
sigmas_krr=np.array([0.3,0.2,0.1,0.075,0.05])
sigmas_krr=np.array([0.2,0.1,0.05])
r2_tr_krr={}
r2_val_krr={}
for sigma_krr in sigmas_krr:
  r2_tr_krr[sigma_krr]=[]
  r2_val_krr[sigma_krr]=[]


N_VAL=200
N_SEEDS=100
N_PERS=np.arange(1,9)
#N_PERS=np.arange(1,6)

labs = ['KGD']
lines=[Line2D([0],[0],color='C0',lw=3)]
for c, sigma_krr in enumerate(sigmas_krr):
  lines.append(Line2D([0],[0],color='C'+str(c+1),lw=3))
  labs.append('$\\sigma='+str(sigma_krr)+'$')

r2_tr_seed=[]
r2_val_seed=[]
r2_tr_seed_krr={}
r2_val_seed_krr={}
for sigma_krr in sigmas_krr:
  r2_tr_seed_krr[sigma_krr]=[]
  r2_val_seed_krr[sigma_krr]=[]

for seed in range(N_SEEDS):
  r2_tr_seed=[]
  r2_val_seed=[]
  r2_tr_seed_krr={}
  r2_val_seed_krr={}
  for sigma_krr in sigmas_krr:
    r2_tr_seed_krr[sigma_krr]=[]
    r2_val_seed_krr[sigma_krr]=[]
  for n_per in N_PERS:
    np.random.seed(seed)
    n_tr=20*n_per
    x_tr=np.random.uniform(-1,1,n_tr).reshape((-1,1))
    y_tr=f(x_tr,n_per)+np.random.normal(0,.2,x_tr.shape)
    x_val=np.random.uniform(-1,1,N_VAL).reshape((-1,1))
    y_val=f(x_val,n_per)#+np.random.normal(0,0,x_val.shape)
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
      gd_obj=gd_alg(Ks,y_tr,'gd','pred',step_size, var0=y1)
      if np.max(np.abs(y1))<1000:
        gd_obj.gd_step()
      y1=gd_obj.get_fs()
      #sigma=(sigma+sigma_m*alpha*step_size)/(1+alpha*step_size)
      sigma=(sigma-sigma_m**2*alpha*step_size)/(1+alpha*step_size*(sigma-2*sigma_m))
    #axs1.cla()
    #axs1.plot(x_tr,y_tr,'ok')
    #axs1.plot(x_val,y_val,'xr')
    #axs1.plot(xs[xs_argsort,0],y1[xs_argsort,0])
    #axs1.set_ylim([-2,2])
    #fig1.savefig('figures/sweep_bw_b.pdf')
    #sleep(.1)
    r2_tr_seed.append(r2(y_tr,y1[:n_tr,:]))
    r2_val_seed.append(r2(y_val,y1[n_tr:,:]))
  
  axs_tr.cla()
  axs_val.cla()
  
  r2_tr.append(r2_tr_seed)
  r2_val.append(r2_val_seed)
  r2_tr_med=np.median(np.array(r2_tr),0)
  r2_val_med=np.median(np.array(r2_val),0)
  r2_tr_lq=np.quantile(np.array(r2_tr),0.25,0)
  r2_tr_uq=np.quantile(np.array(r2_tr),0.75,0)
  r2_val_lq=np.quantile(np.array(r2_val),0.25,0)
  r2_val_uq=np.quantile(np.array(r2_val),0.75,0)
  axs_tr.plot(N_PERS,r2_tr_med,'C0')
  axs_tr.plot(N_PERS,r2_tr_lq,'C0--')
  axs_tr.plot(N_PERS,r2_tr_uq,'C0--')
  axs_tr.set_xlabel('Periods')
  axs_tr.set_ylabel('Training $R^2$')
  axs_val.plot(N_PERS,r2_val_med,'C0')
  axs_val.plot(N_PERS,r2_val_lq,'C0--')
  axs_val.plot(N_PERS,r2_val_uq,'C0--')
  axs_val.set_xlabel('Periods')
  axs_val.set_ylabel('Validation $R^2$')
  
  print(seed)
  r2_tr_med_krr={}
  r2_val_med_krr={}
  r2_tr_lq_krr={}
  r2_val_lq_krr={}
  r2_tr_uq_krr={}
  r2_val_uq_krr={}
  for c,sigma_krr in enumerate(sigmas_krr):
    r2_tr_krr[sigma_krr].append(r2_tr_seed_krr[sigma_krr])
    r2_val_krr[sigma_krr].append(r2_val_seed_krr[sigma_krr])
    r2_tr_med_krr[sigma_krr]=np.median(np.array(r2_tr_krr[sigma_krr]),0)
    r2_val_med_krr[sigma_krr]=np.median(np.array(r2_val_krr[sigma_krr]),0)
    r2_tr_lq_krr[sigma_krr]=np.quantile(np.array(r2_tr_krr[sigma_krr]),0.25,0)
    r2_tr_uq_krr[sigma_krr]=np.quantile(np.array(r2_tr_krr[sigma_krr]),0.75,0)
    r2_val_lq_krr[sigma_krr]=np.quantile(np.array(r2_val_krr[sigma_krr]),0.25,0)
    r2_val_uq_krr[sigma_krr]=np.quantile(np.array(r2_val_krr[sigma_krr]),0.75,0)
    axs_tr.plot(N_PERS,r2_tr_med_krr[sigma_krr],'C'+str(c+1))
    axs_tr.plot(N_PERS,r2_tr_lq_krr[sigma_krr],'C'+str(c+1)+'--')
    axs_tr.plot(N_PERS,r2_tr_uq_krr[sigma_krr],'C'+str(c+1)+'--')
    axs_val.plot(N_PERS,r2_val_med_krr[sigma_krr],'C'+str(c+1))
    axs_val.plot(N_PERS,r2_val_lq_krr[sigma_krr],'C'+str(c+1)+'--')
    axs_val.plot(N_PERS,r2_val_uq_krr[sigma_krr],'C'+str(c+1)+'--')
  axs_tr.set_ylim([-0.01,1.01])
  fig_tr.legend(lines, labs, loc='lower center', ncol=len(labs))
  fig_tr.tight_layout()
  fig_tr.subplots_adjust(bottom=.2)
  fig_tr.savefig('figures/sweep_bw_tr_g1.pdf')
  axs_val.set_ylim([-0.01,1.01])
  fig_val.legend(lines, labs, loc='lower center', ncol=len(labs))
  fig_val.tight_layout()
  fig_val.subplots_adjust(bottom=.2)
  fig_val.savefig('figures/sweep_bw_val_g1.pdf')
