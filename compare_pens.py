import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import markers
from gd_algs import gd_alg, prox_grad
from scipy.linalg import expm

def kern_gauss(X1,X2,sigma):
  if X1.shape[1]==1 and X2.shape[1]==1:
    return np.exp(-0.5*np.square((X1-X2.T)/sigma))
  X1X1=np.sum(np.square(X1),1).reshape((-1,1))
  X2X2=np.sum(np.square(X2),1).reshape((-1,1))
  X1X2=X1@X2.T
  D2=X1X1-2*X1X2+X2X2.T
  return np.exp(-0.5*D2/sigma**2)

def pen_reg(K, Ks, y, lbda, nrm, space):
    #if space=='orac': space='pred'
    K_tr=K if space=='par' else Ks
    prox_obj=prox_grad(K_tr,y,lbda,nrm,space)
    var_old=np.ones(K_tr.shape[0])
    for ii in range(100000):
      prox_obj.prox_step()
      var=prox_obj.get_var()
      if np.allclose(var_old, var, rtol=0.0001, atol=0.0001):
        break
      var_old=np.copy(var)
    return (Ks@var, var) if space=='par' else (var, np.zeros(K.shape[1]))
  
def gd_reg(K, Ks, y, t, alg, space, y1_lbda):
    mse_max=100
    step_size=0.0001
    if space=='par':
      K_tr=K 
    elif space=='pred':
      K_tr=Ks
    #elif space=='orac':
    #  K_tr=Kss
    #y_tr=Ks@np.linalg.solve(K,y) if space=='orac' else y
    gd_obj=gd_alg(K_tr,y,alg,space,step_size)
    best_mse=np.inf
    best_y1=np.zeros((Ks.shape[0],1))
    best_alpha=np.zeros((Ks.shape[1],1))
    alpha=best_alpha
    mse_counter=0
    while 1:
      gd_obj.gd_step()
      if space=='par':
        alpha=gd_obj.get_alpha()
        y1= Ks@alpha
      else:
        y1=gd_obj.get_fs()
      #mse=np.mean(np.abs(y1-y1_lbda))
      if space =='pred' and alg=='sgd':
        mse=np.mean((np.max(np.abs(y1))-np.max(np.abs(y1_lbda)))**2)
      else:
        mse=np.mean((y1-y1_lbda)**2)
      mse_counter+=1
      if mse<best_mse:
        best_mse=mse
        best_y1=y1
        best_alpha=alpha
        mse_counter=0
      if mse_counter>mse_max:
        break
    return best_y1, best_alpha

def f(x):
  y=np.sin(np.pi*x)
  return y

N=1001
l=3
seed=0

np.random.seed(seed)
#x=np.sort(np.random.uniform(0,l,n).reshape((-1,1)),0)
x=np.array([0.1,0.5,1.0,1.5,2.1,2.4,2.6,2.9]).reshape((-1,1))
n=x.shape[0]
y=f(x)+np.random.normal(0,.01,x.shape)
y[1]+=3
x1=np.linspace(0,l,N).reshape((-1,1))
xs=np.vstack((x,x1))
xs_argsort=xs.argsort(0)
sigma=0.3
K=kern_gauss(x,x,sigma)
Ks=kern_gauss(xs,x,sigma)
#Kss=kern_gauss(xs,xs,sigma)
alphah0=np.linalg.solve(K,y)
fsh0=Ks@alphah0

fig1,axs1=plt.subplots(4,3,figsize=(10,8))
fig2,axs2=plt.subplots(4,3,figsize=(10,8))
#fig3,axs3=plt.subplots(4,3,figsize=(99,6))
fig_a,axs_a=plt.subplots(4,3,figsize=(10,8))
for axs in (axs1, axs2, axs_a):
  axs[0,0].set_title('$\\ell_2$ and Gradient Flow')
  axs[0,1].set_title('$\\ell_\\inf$ and Sign Gradient Descent')
  axs[0,2].set_title('$\\ell_1$ and Coordinate Descent')

labs=['Observed Data','Penalized Solution', 'Early Stopping Solution', 'Non- and Fully Reguralized Solutions']
labs_a=['Penalized Solution', 'Early Stopping Solution', 'Non-reguralized Solution']
lines=[plt.plot(0,0,'ok')[0]]
plt.cla()
for c in ['C0','C2','C7']:
  lines.append(Line2D([0],[0],color=c,lw=2))
lines_a=[]
plt.cla()
for c in ['C0','C2','silver']:
  lines_a.append(Line2D([0],[0],color=c,lw=2))


dict_lbda={'lbda_pred': {'l2': [], 'linf': [], 'l1': []}, 'lbda_par': {'l2': [], 'linf': [], 'l1': []}}
dict_lbda['lbda_pred']['l2']=[4,2, 1, .3]
dict_lbda['lbda_pred']['linf']=[1.,0.5, 0.25, 0.01]
dict_lbda['lbda_pred']['l1']=[3e-3,1e-3, 2e-4, 1e-4]

dict_lbda['lbda_par']['l2']=[4,2, 1, .3]
dict_lbda['lbda_par']['linf']=[5e-3, 4e-3, 2.5e-3, 3e-4]
dict_lbda['lbda_par']['l1']=[3e-3,1e-3, 7e-4, 3e-4]

#dict_lbda['lbda_orac']= dict_lbda['lbda_pred']

BW=0.35
LW=2
for space,fig,axs in zip(['pred','par'],[fig1,fig2],[axs1,axs2]):
  if space=='par':
    fig.suptitle('Parameter Space')
    fig_a.suptitle('Parameter Space, alpha')
    fig_a.legend(lines_a, labs_a, loc='lower center', ncol=3)
    fig_a.tight_layout()
    fig_a.subplots_adjust(bottom=0.08)
  else:
    fig.suptitle('Prediction Space')
  fig.legend(lines, labs, loc='lower center', ncol=4)
  fig.tight_layout()
  fig.subplots_adjust(bottom=0.08)
  for a_r in range(axs.shape[0]):
    for a_c, (nrm, alg) in enumerate(zip(['l2','linf','l1'], ['gd','sgd','cd'])):
      axs[a_r,a_c].plot(x,y,'ok')
      axs[a_r,a_c].plot(xs[xs_argsort,0],fsh0[xs_argsort,0],'C7')
      axs[a_r,a_c].plot(xs, np.zeros(xs.shape),'C7')
      if nrm=='l2':
        lbda=dict_lbda['lbda_'+space][nrm][a_r]
        t=1/lbda
        alpha_lbda=np.linalg.solve(K+lbda*np.eye(n),y)
        alpha_t=(np.eye(n)-expm(-t*K))@np.linalg.inv(K)@y
        fsh_lbda=Ks@alpha_lbda
        fsh_t=Ks@alpha_t
      else:
        lbda=dict_lbda['lbda_'+space][nrm][a_r]
        fsh_lbda, alpha_lbda=pen_reg(K,Ks,y, lbda, nrm, space)
        fsh_t, alpha_t=gd_reg(K,Ks,y,t,alg,space, fsh_lbda)
      axs[a_r,a_c].plot(xs[xs_argsort,0],fsh_lbda[xs_argsort,0],'C0',ls=(0,[3,2]),lw=LW)
      axs[a_r,a_c].plot(xs[xs_argsort,0],fsh_t[xs_argsort,0],'C2',ls=(0,[2,2]),lw=LW)
      fig.savefig('figures/compare_pen_'+space+'.pdf')
      if space=='par':
        axs_a[a_r,a_c].bar(np.arange(n),np.squeeze(alphah0),color='silver', width=2*BW)
        axs_a[a_r,a_c].bar(np.arange(n)-BW/2,np.squeeze(alpha_lbda),color='C0', width=BW)
        axs_a[a_r,a_c].bar(np.arange(n)+BW/2,np.squeeze(alpha_t),color='C2', width=BW)
        fig_a.savefig('figures/compare_pen_'+space+'_alpha.pdf')

