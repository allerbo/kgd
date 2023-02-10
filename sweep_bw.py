import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import markers
from gd_algs import gd_alg
from time import sleep
import sys
from gd_algs import kgd

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

def f(x,freq):
  return np.sin(freq*2*np.pi*x)



step_size=0.1

sigma_0=2
sigma_m=1e-5
alpha=20
t=1e2
lbda=1e-4
gamma=1-lbda*t

step_size1=0.01
t1=1e2
gamma1=1-lbda*t1
gamma1=.999


fig,axs=plt.subplots(2,1,figsize=(20,8))
fig_expl,axs_expl=plt.subplots(2,4,figsize=(20,8))
sigmas_krr=np.array([5,0.5,0.1,0.01])

labs = ['True Function', 'Observed Data','KGD', 'KRR']

lines=[Line2D([0],[0],color='C7',lw=3),plt.plot(0,0,'ok')[0],Line2D([0],[0],color='C2',lw=3),Line2D([0],[0],color='C1',lw=3)]
plt.cla()

seed=0
x1=np.linspace(-1,1,301).reshape((-1,1))
for axs_row, freq in zip([axs_expl[0,:],axs_expl[1,:]],[1,6]):
  np.random.seed(seed)
  n_tr=20*freq
  x_tr=np.random.uniform(-1,1,n_tr).reshape((-1,1))
  y_tr=f(x_tr,freq)+np.random.normal(0,.2,x_tr.shape)
  y_true=f(x1,freq)
  xs=np.vstack((x_tr,x1))
  xs_argsort=xs.argsort(0)
  
  y1=kgd(xs,x_tr,y_tr,t,step_size=step_size,alpha=alpha,gamma=gamma)
  for ax, sigma_krr in zip(axs_row,sigmas_krr):
    if freq==1:
      ax.set_title('$\\sigma='+str(sigma_krr)+'$')
    y1_krr=kern_gauss(xs,x_tr,sigma_krr)@np.linalg.solve(kern_gauss(x_tr,x_tr,sigma_krr)+lbda*np.eye(n_tr),y_tr)
    #y1_kgd=kgd(xs,x_tr,y_tr,t,step_size=step_size1,sigma0=sigma_krr,gamma=gamma1)
    ax.plot(x_tr,y_tr,'ok')
    ax.plot(x1, y_true, 'C7')
    ax.plot(xs[xs_argsort,0],y1[xs_argsort,0],'C2')
    ax.plot(xs[xs_argsort,0],y1_krr[xs_argsort,0], 'C1')
    #ax.plot(xs[xs_argsort,0],y1_kgd[xs_argsort,0], 'C3--')
    ax.set_ylim([-1.6,1.6])
    axs_row[0].set_ylabel('$f='+str(freq)+'$')
    fig_expl.legend(lines, labs, loc='lower center', ncol=len(labs))
    fig_expl.tight_layout()
    fig_expl.subplots_adjust(bottom=.08)
    fig_expl.savefig('figures/sweep_bw_expl.pdf')

#sys.exit()

r2_tr=[]
r2_val=[]

sigmas_krr=np.array([0.5,0.2,0.1])
r2_tr_krr={}
r2_val_krr={}
for sigma_krr in sigmas_krr:
  r2_tr_krr[sigma_krr]=[]
  r2_val_krr[sigma_krr]=[]


N_VAL=200
N_SEEDS=100
FREQS=np.arange(1,9)

sigma_cs=['C0','C1','C3']
labs = ['KGD']
lines=[Line2D([0],[0],color='C2',lw=3)]
for sigma_c, sigma_krr in zip(sigma_cs, sigmas_krr):
  lines.append(Line2D([0],[0],color=sigma_c,lw=3))
  labs.append('KRR, $\\sigma='+str(sigma_krr)+'$')

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
  for freq in FREQS:
    np.random.seed(seed)
    n_tr=20*freq
    x_tr=np.random.uniform(-1,1,n_tr).reshape((-1,1))
    y_tr=f(x_tr,freq)+np.random.normal(0,.2,x_tr.shape)
    x_val=np.random.uniform(-1,1,N_VAL).reshape((-1,1))
    y_val=f(x_val,freq)#+np.random.normal(0,0,x_val.shape)
    for sigma_krr in sigmas_krr:
      y1_tr_krr=kern_gauss(x_tr,x_tr,sigma_krr)@np.linalg.solve(kern_gauss(x_tr,x_tr,sigma_krr)+lbda*np.eye(n_tr),y_tr)
      y1_val_krr=kern_gauss(x_val,x_tr,sigma_krr)@np.linalg.solve(kern_gauss(x_tr,x_tr,sigma_krr)+lbda*np.eye(n_tr),y_tr)
      r2_tr_seed_krr[sigma_krr].append(r2(y_tr,y1_tr_krr))
      r2_val_seed_krr[sigma_krr].append(r2(y_val,y1_val_krr))
    
    xs=np.vstack((x_tr,x_val))
    xs_argsort=xs.argsort(0)
    y1=kgd(xs,x_tr,y_tr,t,step_size=step_size,alpha=alpha,gamma=gamma)
   # y1=np.zeros(xs.shape)
   # sigma=sigma_0
   # for i in range(int(t/step_size)):
   #   Ks=kern_gauss(xs,x_tr,sigma)
   #   gd_obj=gd_alg(Ks,y_tr,'gd','pred',step_size, var0=y1)
   #   if np.max(np.abs(y1))<1000:
   #     gd_obj.gd_step()
   #   y1=gd_obj.get_fs()
   #   #sigma=(sigma+sigma_m*alpha*step_size)/(1+alpha*step_size)
   #   #sigma=(sigma-sigma_m**2*alpha*step_size)/(1+alpha*step_size*(sigma-2*sigma_m))
   #   sigma=sigma/(1+alpha*step_size*sigma)
    #axs1.cla()
    #axs1.plot(x_tr,y_tr,'ok')
    #axs1.plot(x_val,y_val,'xr')
    #axs1.plot(xs[xs_argsort,0],y1[xs_argsort,0])
    #axs1.set_ylim([-2,2])
    #fig1.savefig('figures/sweep_bw_b.pdf')
    #sleep(.1)
    r2_tr_seed.append(r2(y_tr,y1[:n_tr,:]))
    r2_val_seed.append(r2(y_val,y1[n_tr:,:]))
  
  axs[0].cla()
  axs[1].cla()
  
  r2_tr.append(r2_tr_seed)
  r2_val.append(r2_val_seed)
  r2_tr_med=np.median(np.array(r2_tr),0)
  r2_val_med=np.median(np.array(r2_val),0)
  r2_tr_lq=np.quantile(np.array(r2_tr),0.25,0)
  r2_tr_uq=np.quantile(np.array(r2_tr),0.75,0)
  r2_val_lq=np.quantile(np.array(r2_val),0.25,0)
  r2_val_uq=np.quantile(np.array(r2_val),0.75,0)
  axs[0].plot(FREQS,r2_tr_med,'C2')
  axs[0].plot(FREQS,r2_tr_lq,'C2--')
  axs[0].plot(FREQS,r2_tr_uq,'C2--')
  axs[0].set_xlabel('Frequency')
  axs[0].set_ylabel('Training $R^2$')
  axs[1].plot(FREQS,r2_val_med,'C2')
  axs[1].plot(FREQS,r2_val_lq,'C2--')
  axs[1].plot(FREQS,r2_val_uq,'C2--')
  axs[1].set_xlabel('Frequency')
  axs[1].set_ylabel('Validation $R^2$')
  
  print(seed)
  r2_tr_med_krr={}
  r2_val_med_krr={}
  r2_tr_lq_krr={}
  r2_val_lq_krr={}
  r2_tr_uq_krr={}
  r2_val_uq_krr={}
  for sigma_c,sigma_krr in zip(sigma_cs,sigmas_krr):
    r2_tr_krr[sigma_krr].append(r2_tr_seed_krr[sigma_krr])
    r2_val_krr[sigma_krr].append(r2_val_seed_krr[sigma_krr])
    r2_tr_med_krr[sigma_krr]=np.median(np.array(r2_tr_krr[sigma_krr]),0)
    r2_val_med_krr[sigma_krr]=np.median(np.array(r2_val_krr[sigma_krr]),0)
    r2_tr_lq_krr[sigma_krr]=np.quantile(np.array(r2_tr_krr[sigma_krr]),0.25,0)
    r2_tr_uq_krr[sigma_krr]=np.quantile(np.array(r2_tr_krr[sigma_krr]),0.75,0)
    r2_val_lq_krr[sigma_krr]=np.quantile(np.array(r2_val_krr[sigma_krr]),0.25,0)
    r2_val_uq_krr[sigma_krr]=np.quantile(np.array(r2_val_krr[sigma_krr]),0.75,0)
    axs[0].plot(FREQS,r2_tr_med_krr[sigma_krr],sigma_c)
    axs[0].plot(FREQS,r2_tr_lq_krr[sigma_krr],sigma_c+'--')
    axs[0].plot(FREQS,r2_tr_uq_krr[sigma_krr],sigma_c+'--')
    axs[1].plot(FREQS,r2_val_med_krr[sigma_krr],sigma_c)
    axs[1].plot(FREQS,r2_val_lq_krr[sigma_krr],sigma_c+'--')
    axs[1].plot(FREQS,r2_val_uq_krr[sigma_krr],sigma_c+'--')
  axs[0].set_ylim([-0.01,1.01])
  axs[1].set_ylim([-0.01,1.01])
  fig.legend(lines, labs, loc='lower center', ncol=len(labs))
  fig.tight_layout()
  fig.subplots_adjust(bottom=.1)
  fig.savefig('figures/sweep_bw.pdf')
