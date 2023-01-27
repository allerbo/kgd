import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import markers
from gd_algs import gd_alg
from time import sleep
import sys
from sklearn.gaussian_process.kernels import Matern
from scipy.special import kv, gamma

def kern_gauss(X1,X2,sigma):
  if X1.shape[1]==1 and X2.shape[1]==1:
    return np.exp(-0.5*np.square((X1-X2.T)/sigma))
  X1X1=np.sum(np.square(X1),1).reshape((-1,1))
  X2X2=np.sum(np.square(X2),1).reshape((-1,1))
  X1X2=X1@X2.T
  D2=X1X1-2*X1X2+X2X2.T
  return np.exp(-0.5*D2/sigma**2)

def f(x):
  y=np.sin(2*np.pi*x)
  return y

def r2(y,y_hat):
  return 1-np.mean((y-y_hat)**2)/np.mean((y-np.mean(y))**2)

def mse(y,y_hat):
  return np.mean((y-y_hat)**2)


N=1001
l=2

seed=5
#seed=4
#seed=10
#seed=7


np.random.seed(seed)
n_tr=50
n_val=100
#x=np.sort(np.random.uniform(-l,l,n).reshape((-1,1)),0)
#x_tr=np.sort(np.random.uniform(-l,l,n).reshape((-1,1)),0)
x_tr=np.random.normal(0,2,n_tr).reshape((-1,1))
#x_tr=np.random.standard_cauchy(n).reshape((-1,1))
y_tr=f(x_tr)#+np.random.normal(0,.3,x_tr.shape)
#x_val=np.sort(np.random.uniform(-l,l,n).reshape((-1,1)),0)
#x_val=np.random.standard_cauchy(n).reshape((-1,1))
x_val=np.random.normal(0,2,n_val).reshape((-1,1))
y_val=f(x_val)#+np.random.normal(0,.0,x_val.shape)
x1=np.linspace(-l,l,N).reshape((-1,1))
xs=np.vstack((x_tr,x_val))
#xs=np.vstack((x_tr,x1))
xs_argsort=xs.argsort(0)




fig,axs=plt.subplots(1,1,figsize=(12,6))
#axs.plot(x_tr,y_tr,'ok')
#axs.plot(x_val,y_val,'ok',fillstyle='none')
#axs.plot(x1,f(x1))
#fig.savefig('figures/double_descent.pdf')

#sys.exit()

#np.linalg.solve(kern_mat(x,x,sigma,nu),y)
#y1=kern_gauss(xs,x,sigma)@np.linalg.solve(kern_gauss(x,x,sigma),y)
#y1=kern_mat(xs,x,sigma,nu)@np.linalg.solve(kern_mat(x,x,sigma,nu)+100*np.eye(n),y)

step_size=0.01

alpha=1
#sigma_m=0.5
#sigma_m=0
#sigma_m=1
#sigma_m=0
sigma_ms=[0]
r2_tr=[]
r2_val=[]
#alphas=[0.0001,0.0003, 0.001, 0.003, 0.01,0.03, 0.1,0.3, 1,3,10,30]
#alphas=[.001]
sigma_0=3
sigma_ms=np.geomspace(sigma_0,0.1,100)

sigma_m=0
alpha=1
#for alpha in alphas:
for sigma_m in sigma_ms:
  y1=np.zeros(xs.shape)
  sigma=sigma_0
  n_iter=0
  #sigma_m=0.5
  for i in range(10000):
  #while sigma>1e-10:
    n_iter+=1
    Ks=kern_gauss(xs,x_tr,sigma)
    gd_obj=gd_alg(Ks,y_tr,'gd','pred',step_size, var0=y1)
    gd_obj.gd_step()
    y1=gd_obj.get_fs()
    sigma=(sigma+sigma_m*alpha*step_size)/(1+alpha*step_size)
    #if n_iter % 1000==0:
    #  print(sigma,r2(y_tr,y1[:n_tr,:]),r2(y_val,y1[n_tr:,:]))
    #  #print(sigma)
    #  axs.cla()
    #  axs.plot(x_tr,y_tr,'ok')
    #  axs.plot(xs[xs_argsort,0],y1[xs_argsort,0])
    #  #axs.plot(x_val,y_val,'vk')
    #  #axs.plot(x_tr,y1[:n_tr,:],'ok',fillstyle='none')
    #  #axs.plot(x_val,y1[n_tr:,:],'vk',fillstyle='none')
    #  fig.savefig('figures/double_descent1.pdf')
    #  sleep(.1)
  #print(sigma)
  print(alpha,sigma,r2(y_tr,y1[:n_tr,:]),r2(y_val,y1[n_tr:,:]))
  #r2_tr.append(r2(y_tr,y1[:n_tr,:]))
  #r2_val.append(r2(y_val,y1[n_tr:,:]))
  r2_tr.append(mse(y_tr,y1[:n_tr,:]))
  r2_val.append(mse(y_val,y1[n_tr:,:]))
  axs.cla()
  axs.plot(1/np.array(sigma_ms[:len(r2_tr)]),r2_tr)
  axs.plot(1/np.array(sigma_ms[:len(r2_val)]),r2_val)
  fig.savefig('figures/double_descent1.pdf')

