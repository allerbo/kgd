import numpy as np
import sys
from gd_algs import kgd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import markers
import sys
KGDC=False

def krr(xs,x_tr,y_tr,lbda,sigma):
  Ks=np.exp(-0.5*np.square((xs-x_tr.T)/sigma))
  K=np.exp(-0.5*np.square((x_tr-x_tr.T)/sigma))
  return Ks@np.linalg.solve(K+lbda*np.eye(K.shape[0]),y_tr)

def r2(y,y_hat):
  return 1-np.mean((y-y_hat)**2)/np.mean((y-np.mean(y))**2)

def mse(y,y_hat):
  return np.mean((y-y_hat)**2)


def plot_med_q(ax,x,arr,c,QUANT=0.25):
  ax.plot(x,np.squeeze(np.median(arr,0)),c)
  ax.plot(x,np.squeeze(np.quantile(arr,QUANT,0)),c+'--')
  ax.plot(x,np.squeeze(np.quantile(arr,1-QUANT,0)),c+'--')

def f1(x,freq):
  return np.sin(freq*2*np.pi*x)

def f2(x,freqs):
  y=np.zeros(x.shape)
  y+=np.sin(freqs[0]*2*np.pi*x)*((x>-2)&(x<0))
  y+=np.sin(freqs[1]*2*np.pi*x)*((x>0)&(x<2))
  return y



SWEEP_FREQS= np.arange(1,9) 
SWEEP_FREQS_EXPL=[1,6]
ONE_FREQ=[1]
TWO_FREQS=[1,7]

OBS_FREQ=20
N_VAL=200

step_size=0.1
alpha=20
t=1e4
lbda=1/t


TYPE='double_descent'
TYPE='two_freqs'
TYPE='sweep_freqs'


for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

if TYPE=='sweep_freqs':
  fig_expl,axs_expl=plt.subplots(2,4,figsize=(20,8))
  axs_expl=[axs_expl[0,:],axs_expl[1,:]]
  SIGMAS_EXPL=[np.array([5,0.5,0.1,0.01]),np.array([5,0.5,0.1,0.01])]
  SEED_EXPL=0
  FREQS_EXPL=SWEEP_FREQS_EXPL
  
  FREQS=SWEEP_FREQS
  fig,axs=plt.subplots(2,1,figsize=(20,8))
  SIGMAS=np.array([0.5,0.2,0.1])
  SIGMA_CS=['C0','C1','C3']
  labs = ['Decreasing']
  lines=[Line2D([0],[0],color='C2',lw=3)]
  for sigma_c, sigma in zip(SIGMA_CS, SIGMAS):
    lines.append(Line2D([0],[0],color=sigma_c,lw=3))
    labs.append('Constant, $\\sigma='+str(sigma)+'$')
else:
  FREQS=ONE_FREQ
  FREQS_EXPL=ONE_FREQ
  fig,ax_big=plt.subplots(1,1,figsize=(14,6))
  labs = ['Decreasing, Training Error', 'Constant, Training Error', 'Decreasing, Validation Error', 'Constant, Validation Error']
  lines=[Line2D([0],[0],color='C0',lw=3),Line2D([0],[0],color='C1',lw=3),Line2D([0],[0],color='C2',lw=3),Line2D([0],[0],color='C3',lw=3)]
  
  if TYPE=='double_descent':
    SIGMAS=np.geomspace(2,1e-2,100)
    fig_expl,axs_expl=plt.subplots(3,2,figsize=(14,12))
    SIGMAS_EXPL=[np.array([0.01, 0.03, 0.08, 0.6, 1,5])]
    SEED_EXPL=6
    SEED_EXPL=4
    SEED_EXPL=3
  elif TYPE=='two_freqs':
    SIGMAS=np.geomspace(50,1e-4,100)
    fig_expl,axs_expl=plt.subplots(4,2,figsize=(14,12))
    SIGMAS_EXPL=[np.array([0.0001, 0.002, 0.01, 0.04, 0.15,0.6,50,0])]
    SEED_EXPL=0
  axs_expl=[axs_expl]

labs_expl = ['True Function', 'Observed Data','Decreasing', 'Constant']
lines_expl=[Line2D([0],[0],color='C7',lw=3),plt.plot(0,0,'ok')[0],Line2D([0],[0],color='C2',lw=3),Line2D([0],[0],color='C1',lw=3)]
plt.cla()

for freq, sigmas, axs in zip(FREQS_EXPL, SIGMAS_EXPL, axs_expl):
  np.random.seed(SEED_EXPL)
  if TYPE=='two_freqs':
    x_tr=np.hstack((np.random.uniform(-2,0,OBS_FREQ*TWO_FREQS[0]),np.random.uniform(0,2,OBS_FREQ*TWO_FREQS[1]))).reshape((-1,1))
    y_tr=f2(x_tr,TWO_FREQS)+np.random.normal(0,.2,x_tr.shape)
    x1=np.linspace(-2,2,301).reshape((-1,1))
    y_true=f2(x1,TWO_FREQS)
  else:
    x_tr=np.random.uniform(-1,1,OBS_FREQ*freq).reshape((-1,1))
    y_tr=f1(x_tr,freq)+np.random.normal(0,.2,x_tr.shape)
    x1=np.linspace(-1,1,301).reshape((-1,1))
    y_true=f1(x1,freq)
  
  xs=np.vstack((x_tr,x1))
  xs_argsort=xs.argsort(0)
  
  for sigma, ax in zip(sigmas, axs.ravel()):
    ax.plot(x_tr,y_tr,'ok')
    ax.plot(x1, y_true, 'C7')
    
    if TYPE=='two_freqs' and sigma==0:
      ax.set_title('Decreasing')
      y1_kgdd,tt=kgd(xs,x_tr,y_tr,t,step_size=step_size,alpha=alpha)
      ax.plot(xs[xs_argsort,0],y1_kgdd[xs_argsort,0],'C2')
    elif TYPE=='double_descent':
      y1_kgdd,tt=kgd(xs,x_tr,y_tr,t,step_size=step_size,alpha=alpha, sigma_min=sigma)
      ax.plot(xs[xs_argsort,0],y1_kgdd[xs_argsort,0],'C2')
    elif TYPE=='sweep_freqs':
      y1_kgdd,tt=kgd(xs,x_tr,y_tr,t,step_size=step_size,alpha=alpha)
      ax.plot(xs[xs_argsort,0],y1_kgdd[xs_argsort,0],'C2')
    if not(TYPE=='two_freqs' and sigma==0):
      if TYPE=='sweep_freqs':
        if freq==1:
          ax.set_title('$\\sigma='+str(sigma)+'$')
      else:
        ax.set_title('Constant, $\\sigma='+str(sigma)+'$')
      y1_krr=krr(xs,x_tr,y_tr,lbda,sigma)
      ax.plot(xs[xs_argsort,0],y1_krr[xs_argsort,0], 'C1')
      if KGDC:
        y1_kgdc,tt=kgd(xs,x_tr,y_tr,t,step_size=step_size,sigma0=sigma)
        ax.plot(xs[xs_argsort,0],y1_kgdc[xs_argsort,0], 'C3')
    ax.set_ylim([-1.5,1.5])
  
  fig_expl.legend(lines_expl, labs_expl, loc='lower center', ncol=len(labs))
  fig_expl.tight_layout()
  fig_expl.subplots_adjust(bottom=.08)
  fig_expl.savefig('figures/'+TYPE+'_expl.pdf')

#sys.exit()










SEEDS=range(100)


if TYPE=='double_descent':
  r2_tr_kgdd=np.zeros([len(SEEDS), len(FREQS), len(SIGMAS)])
  r2_val_kgdd=np.zeros([len(SEEDS), len(FREQS), len(SIGMAS)])
else:
  r2_tr_kgdd=np.zeros([len(SEEDS), len(FREQS), 1])
  r2_val_kgdd=np.empty([len(SEEDS), len(FREQS), 1])

r2_tr_kgdc=np.zeros([len(SEEDS), len(FREQS), len(SIGMAS)])
r2_val_kgdc=np.zeros([len(SEEDS), len(FREQS), len(SIGMAS)])
r2_tr_krr=np.zeros([len(SEEDS), len(FREQS), len(SIGMAS)])
r2_val_krr=np.zeros([len(SEEDS), len(FREQS), len(SIGMAS)])


for i_seed, seed in enumerate(SEEDS):
  print(seed)
  for i_freq, freq in enumerate(FREQS):
    np.random.seed(seed)
    if TYPE=='two_freqs':
      x_tr=np.hstack((np.random.uniform(-2,0,OBS_FREQ*TWO_FREQS[0]),np.random.uniform(0,2,OBS_FREQ*TWO_FREQS[1]))).reshape((-1,1))
      y_tr=f2(x_tr,TWO_FREQS)+np.random.normal(0,.2,x_tr.shape)
      x_val=np.random.uniform(-2,2,N_VAL).reshape((-1,1))
      y_val=f2(x_val,TWO_FREQS)
    else:
      x_tr=np.random.uniform(-1,1,OBS_FREQ*freq).reshape((-1,1))
      y_tr=f1(x_tr,freq)+np.random.normal(0,.2,x_tr.shape)
      x_val=np.random.uniform(-1,1,N_VAL).reshape((-1,1))
      y_val=f1(x_val,freq)
    
    n_tr=x_tr.shape[0]
    xs=np.vstack((x_tr,x_val))
    xs_argsort=xs.argsort(0)
  
    if TYPE in ['sweep_freqs', 'two_freqs']:
      y1_kgdd,tt=kgd(xs,x_tr,y_tr,t,step_size=step_size,alpha=alpha)
      r2_tr_kgdd[i_seed,i_freq,0]=r2(y_tr,y1_kgdd[:n_tr,:])
      r2_val_kgdd[i_seed,i_freq,0]=r2(y_val,y1_kgdd[n_tr:,:])
    
    for i_sigma, sigma in enumerate(SIGMAS):
      y1_krr=krr(xs,x_tr,y_tr,lbda,sigma)
      r2_tr_krr[i_seed,i_freq,i_sigma]=r2(y_tr,y1_krr[:n_tr,:])
      r2_val_krr[i_seed,i_freq,i_sigma]=r2(y_val,y1_krr[n_tr:,:])
      
      if KGDC:
        y1_kgdc,tt=kgd(xs,x_tr,y_tr,t,step_size=step_size,sigma0=sigma)
        r2_tr_kgdc[i_seed,i_freq,i_sigma]=r2(y_tr,y1_kgdc[:n_tr,:])
        r2_val_kgdc[i_seed,i_freq,i_sigma]=r2(y_val,y1_kgdc[n_tr:,:])
       
      if TYPE=='double_descent':
        y1_kgdd,tt=kgd(xs,x_tr,y_tr,t,step_size=step_size,alpha=alpha, sigma_min=sigma)
        r2_tr_kgdd[i_seed,i_freq,i_sigma]=r2(y_tr,y1_kgdd[:n_tr,:])
        r2_val_kgdd[i_seed,i_freq,i_sigma]=r2(y_val,y1_kgdd[n_tr:,:])
    
  
  if TYPE=='sweep_freqs':
    axs[0].cla()
    axs[1].cla()
    plot_med_q(axs[0],FREQS,r2_tr_kgdd[:(i_seed+1),:,:],'C2',QUANT=0.25)
    plot_med_q(axs[1],FREQS,r2_val_kgdd[:(i_seed+1),:,:],'C2',QUANT=0.25)
    for i_sigma, (sigma_c, sigma) in enumerate(zip(SIGMA_CS,SIGMAS)):
      plot_med_q(axs[0],FREQS,r2_tr_krr[:(i_seed+1),:,i_sigma],sigma_c,QUANT=0.25)
      plot_med_q(axs[1],FREQS,r2_val_krr[:(i_seed+1),:,i_sigma],sigma_c,QUANT=0.25)
      if KGDC:
        plot_med_q(axs[0],FREQS,r2_tr_kgdc[:(i_seed+1),:,i_sigma],sigma_c,QUANT=0.25)
        plot_med_q(axs[1],FREQS,r2_val_kgdc[:(i_seed+1),:,i_sigma],sigma_c,QUANT=0.25)
    axs[0].set_xlabel('Frequency')
    axs[0].set_ylabel('Training $R^2$')
    axs[0].set_ylim([-0.01,1.01])
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Validation $R^2$')
    axs[1].set_ylim([-0.01,1.01])
  else:
    ax_big.cla()
    if TYPE=='two_freqs':
      plot_med_q(ax_big,SIGMAS,np.repeat(r2_tr_kgdd[:(i_seed+1),:,:],len(SIGMAS),2),'C0',QUANT=0.25)
      plot_med_q(ax_big,SIGMAS,np.repeat(r2_val_kgdd[:(i_seed+1),:,:],len(SIGMAS),2),'C2',QUANT=0.25)
    elif TYPE=='double_descent':
      plot_med_q(ax_big,SIGMAS,r2_tr_kgdd[:(i_seed+1),:,:],'C0',QUANT=0.25)
      plot_med_q(ax_big,SIGMAS,r2_val_kgdd[:(i_seed+1),:,:],'C2',QUANT=0.25)
    
    plot_med_q(ax_big,SIGMAS,r2_tr_krr[:(i_seed+1),:,:],'C1',QUANT=0.25)
    plot_med_q(ax_big,SIGMAS,r2_val_krr[:(i_seed+1),:,:],'C3',QUANT=0.25)
    if KGDC:
      plot_med_q(ax_big,SIGMAS,r2_tr_kgdc[:(i_seed+1),:,:],'C1',QUANT=0.25)
      plot_med_q(ax_big,SIGMAS,r2_val_kgdc[:(i_seed+1),:,:],'C3',QUANT=0.25)
    
    for sigma in SIGMAS_EXPL[0]:
      if sigma>0 and sigma!=5:
        ax_big.axvline(sigma,color='k')
    
    ax_big.set_ylim([-0.01,1.01])
    ax_big.set_xscale('log')
          
  fig.legend(lines, labs, loc='lower center', ncol=len(labs))
  fig.tight_layout()
  fig.subplots_adjust(bottom=.1)
  fig.savefig('figures/'+TYPE+'.pdf')

