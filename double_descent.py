import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import markers
#from gd_algs import gd_alg
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

#x1=np.linspace(-2,2,1000).reshape((-1,1))

def f(x,freq):
  y=np.sin(freq*2*np.pi*x)
  return y

#axs.plot(x1,f(x1,ps))
#axs.plot(x_tr,y_tr,'ok')
#axs.plot(x_val,y_val,'or')
#fig.savefig('figures/test.pdf')



L_FI=2
N_PER_FI=20

ps=[1,7]

step_size=0.1
alpha=20
t=1e3
lbda=1e-4
gamma=1-t*lbda

fig,axs=plt.subplots(1,1,figsize=(14,6))
fig_expl,axs_expl=plt.subplots(3,2,figsize=(14,12))
sigmas_krr_expl=np.array([0.01, 0.03, 0.09, 0.51, 2])

labs = ['True Function', 'Observed Data','KGD', 'KRR']

lines=[Line2D([0],[0],color='C7',lw=3),plt.plot(0,0,'ok')[0],Line2D([0],[0],color='C2',lw=3),Line2D([0],[0],color='C1',lw=3)]
plt.cla()

FREQ=1

seed=6
np.random.seed(seed)
x_tr=np.random.uniform(-1,1,N_PER_FI*FREQ).reshape((-1,1))
n_tr=x_tr.shape[0]
y_tr=f(x_tr,FREQ)+np.random.normal(0,.2,x_tr.shape)
x1=np.linspace(-1,1,301).reshape((-1,1))
y_true=f(x1,FREQ)
xs=np.vstack((x_tr,x1))
xs_argsort=xs.argsort(0)
for i, (ax, sigma_krr) in enumerate(zip(axs_expl.ravel(), sigmas_krr_expl)):
  ax.plot(x_tr,y_tr,'ok')
  ax.plot(x1, y_true, 'C7')
  if sigma_krr==0:
    ax.set_title('KGD')
    y1=kgd(xs,x_tr,y_tr,t,step_size=step_size,alpha=alpha,gamma=gamma)
    ax.plot(xs[xs_argsort,0],y1[xs_argsort,0],'C2')
  else:
    ax.set_title('KRR, $\\sigma='+str(sigma_krr)+'$')
    y1_krr=kern_gauss(xs,x_tr,sigma_krr)@np.linalg.solve(kern_gauss(x_tr,x_tr,sigma_krr)+lbda*np.eye(n_tr),y_tr)
    y1_kgd,tt=kgd(xs,x_tr,y_tr,t,step_size=step_size,alpha=alpha,gamma=gamma, sigma_min=sigma_krr)
    #y1_kgd=kgd(xs,x_tr,y_tr,t,step_size=step_size,gamma=gamma,sigma0=sigma_krr)
    ax.plot(xs[xs_argsort,0],y1_krr[xs_argsort,0], 'C1')
    ax.plot(xs[xs_argsort,0],y1_kgd[xs_argsort,0], 'C2')
  ax.set_ylim([-1.5,1.5])
  fig_expl.legend(lines, labs, loc='lower center', ncol=len(labs))
  fig_expl.tight_layout()
  fig_expl.subplots_adjust(bottom=.08)
  fig_expl.savefig('figures/double_descent_expl_hej.pdf')
  
sys.exit()



sigmas_krr=np.geomspace(2,1e-2,20)

r2_tr_seed_kgd={}
r2_val_seed_kgd={}
r2_tr_seed_krr={}
r2_val_seed_krr={}
for sigma_krr in sigmas_krr:
  r2_tr_seed_kgd[sigma_krr]=[]
  r2_val_seed_kgd[sigma_krr]=[]
  r2_tr_seed_krr[sigma_krr]=[]
  r2_val_seed_krr[sigma_krr]=[]

n_val=200

labs = ['KGD Training Error', 'KRR Training Error', 'KGD Validation Error', 'KRR Validation Error']
lines=[Line2D([0],[0],color='C0',lw=3),Line2D([0],[0],color='C1',lw=3),Line2D([0],[0],color='C2',lw=3),Line2D([0],[0],color='C3',lw=3)]

for seed in range(100):
  np.random.seed(seed)
  x_tr=np.random.uniform(-1,1,N_PER_FI*FREQ).reshape((-1,1))
  n_tr=x_tr.shape[0]
  y_tr=f(x_tr,FREQ)+np.random.normal(0,.2,x_tr.shape)
  x_val=np.random.uniform(-1,1,n_val).reshape((-1,1))
  y_val=f(x_val,FREQ)#+np.random.normal(0,0,x_val.shape)
  xs=np.vstack((x_tr,x_val))
  xs_argsort=xs.argsort(0)
  
  for sigma_krr in sigmas_krr:
    y1_tr_krr=kern_gauss(x_tr,x_tr,sigma_krr)@np.linalg.solve(kern_gauss(x_tr,x_tr,sigma_krr)+lbda*np.eye(n_tr),y_tr)
    y1_val_krr=kern_gauss(x_val,x_tr,sigma_krr)@np.linalg.solve(kern_gauss(x_tr,x_tr,sigma_krr)+lbda*np.eye(n_tr),y_tr)
    r2_tr_seed_krr[sigma_krr].append(r2(y_tr,y1_tr_krr))
    r2_val_seed_krr[sigma_krr].append(r2(y_val,y1_val_krr))
     
    y1=kgd(xs,x_tr,y_tr,t,step_size=step_size,alpha=alpha,gamma=gamma, sigma_min=sigma_krr)
    r2_tr_seed_kgd[sigma_krr].append(r2(y_tr,y1[:n_tr,:]))
    r2_val_seed_kgd[sigma_krr].append(r2(y_val,y1[n_tr:,:]))
  
  axs.cla()
 # r2_tr_median=np.median(np.array(r2_tr_seed))
 # r2_tr_lq=np.quantile(np.array(r2_tr_seed),0.25)
 # r2_tr_uq=np.quantile(np.array(r2_tr_seed),0.75)
 # r2_val_median=np.median(np.array(r2_val_seed))
 # r2_val_lq=np.quantile(np.array(r2_val_seed),0.25)
 # r2_val_uq=np.quantile(np.array(r2_val_seed),0.75)
 # 
 # axs.plot(sigmas_krr,len(sigmas_krr)*[r2_tr_median],'C0')
 # axs.plot(sigmas_krr,len(sigmas_krr)*[r2_tr_lq],'C0--')
 # axs.plot(sigmas_krr,len(sigmas_krr)*[r2_tr_uq],'C0--')
 # 
 # axs.plot(sigmas_krr,len(sigmas_krr)*[r2_val_median],'C2')
 # axs.plot(sigmas_krr,len(sigmas_krr)*[r2_val_lq],'C2--')
 # axs.plot(sigmas_krr,len(sigmas_krr)*[r2_val_uq],'C2--')
  
  r2_tr_median_kgds=[]
  r2_tr_lq_kgds=[]
  r2_tr_uq_kgds=[]
  r2_val_median_kgds=[]
  r2_val_lq_kgds=[]
  r2_val_uq_kgds=[]
  
  r2_tr_median_krrs=[]
  r2_tr_lq_krrs=[]
  r2_tr_uq_krrs=[]
  r2_val_median_krrs=[]
  r2_val_lq_krrs=[]
  r2_val_uq_krrs=[]
  for sigma_krr in sigmas_krr:
    r2_tr_median_kgds.append(np.median(np.array(r2_tr_seed_kgd[sigma_krr])))
    r2_tr_lq_kgds.append(np.quantile(np.array(r2_tr_seed_kgd[sigma_krr]),0.25))
    r2_tr_uq_kgds.append(np.quantile(np.array(r2_tr_seed_kgd[sigma_krr]),0.75))
    r2_val_median_kgds.append(np.median(np.array(r2_val_seed_kgd[sigma_krr])))
    r2_val_lq_kgds.append(np.quantile(np.array(r2_val_seed_kgd[sigma_krr]),0.25))
    r2_val_uq_kgds.append(np.quantile(np.array(r2_val_seed_kgd[sigma_krr]),0.75))
    
    r2_tr_median_krrs.append(np.median(np.array(r2_tr_seed_krr[sigma_krr])))
    r2_tr_lq_krrs.append(np.quantile(np.array(r2_tr_seed_krr[sigma_krr]),0.25))
    r2_tr_uq_krrs.append(np.quantile(np.array(r2_tr_seed_krr[sigma_krr]),0.75))
    r2_val_median_krrs.append(np.median(np.array(r2_val_seed_krr[sigma_krr])))
    r2_val_lq_krrs.append(np.quantile(np.array(r2_val_seed_krr[sigma_krr]),0.25))
    r2_val_uq_krrs.append(np.quantile(np.array(r2_val_seed_krr[sigma_krr]),0.75))

  axs.plot(sigmas_krr,r2_tr_median_kgds,'C0')
  axs.plot(sigmas_krr,r2_tr_lq_kgds,'C0--')
  axs.plot(sigmas_krr,r2_tr_uq_kgds,'C0--')

  axs.plot(sigmas_krr,r2_val_median_kgds,'C2')
  axs.plot(sigmas_krr,r2_val_lq_kgds,'C2--')
  axs.plot(sigmas_krr,r2_val_uq_kgds,'C2--')
  
  axs.plot(sigmas_krr,r2_tr_median_krrs,'C1')
  axs.plot(sigmas_krr,r2_tr_lq_krrs,'C1--')
  axs.plot(sigmas_krr,r2_tr_uq_krrs,'C1--')

  axs.plot(sigmas_krr,r2_val_median_krrs,'C3')
  axs.plot(sigmas_krr,r2_val_lq_krrs,'C3--')
  axs.plot(sigmas_krr,r2_val_uq_krrs,'C3--')
  
  for sigma_krr in sigmas_krr_expl:
    if sigma_krr>0:
      axs.axvline(sigma_krr,color='k')
  
  axs.set_ylim([-0.01,1.01])
  axs.set_xscale('log')

  fig.legend(lines, labs, loc='lower center', ncol=len(labs))
  fig.tight_layout()
  fig.subplots_adjust(bottom=.12)
  
  fig.savefig('figures/double_descent.pdf')

