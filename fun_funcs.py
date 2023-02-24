import numpy as np
from kgd import kgd
from matplotlib import pyplot as plt
import sys

def krr(xs,x_tr,y_tr_in,lbda,sigma):
  y_tr_mean=np.mean(y_tr_in)
  y_tr=y_tr_in-y_tr_mean
  Ks=np.exp(-0.5*np.square((xs-x_tr.T)/sigma))
  K=np.exp(-0.5*np.square((x_tr-x_tr.T)/sigma))
  return Ks@np.linalg.solve(K+lbda*np.eye(K.shape[0]),y_tr)+y_tr_mean

def r2(y,y_hat):
  return 1-np.mean((y-y_hat)**2)/np.mean((y-np.mean(y))**2)


NS=500

def f1():
  FREQ=5
  OBS_FREQ=20
  x_tr1=np.random.uniform(-2,-1,FREQ*OBS_FREQ).reshape((-1,1))
  x_tr2=np.random.uniform(1,2,FREQ*OBS_FREQ).reshape((-1,1))
  y_tr1=np.sin(FREQ*2*np.pi*x_tr1)+np.random.normal(0,.1,x_tr1.shape)
  y_tr2=np.sin(FREQ*2*np.pi*x_tr2)+np.random.normal(0,.1,x_tr2.shape)+10
  x_tr=np.vstack((x_tr1,x_tr2))
  y_tr=np.vstack((y_tr1,y_tr2))
  return x_tr, y_tr

def f2():
  FREQ1=1
  FREQ2=10
  OBS_FREQ=30
  x_tr1=np.random.uniform(-2,2,2*FREQ1*OBS_FREQ).reshape((-1,1))
  x_tr2=np.random.uniform(-.1,.1,FREQ2*OBS_FREQ//10).reshape((-1,1))
  y_tr1=np.sin(FREQ1*2*np.pi*x_tr1)+np.random.normal(0,.1,x_tr1.shape)
  y_tr2=np.sin(FREQ2*2*np.pi*x_tr2)+np.random.normal(0,.1,x_tr2.shape)
  x_tr=np.vstack((x_tr1,x_tr2))
  y_tr=np.vstack((y_tr1,y_tr2))
  x_val=np.random.uniform(np.min(x_tr),np.max(x_tr),NS).reshape((-1,1))
  y_val=np.sin(FREQ1*2*np.pi*x_val)*(np.abs(x_val)>0.1)+np.sin(FREQ2*2*np.pi*x_val)*(np.abs(x_val)<0.1)
  return x_tr, y_tr, x_val, y_val

def f2b():
  FREQ1=1
  FREQ2=10
  OBS_FREQ=20
  def fy(x):
    return np.sin(FREQ1*2*np.pi*x)*(np.abs(x)>0.5)+np.sin(FREQ2*2*np.pi*x)*(np.abs(x)<0.5)
  x_tr1=np.random.uniform(-2,2,2*FREQ1*OBS_FREQ).reshape((-1,1))
  x_tr2=np.random.uniform(-.5,.5,FREQ2*OBS_FREQ//2).reshape((-1,1))
  x_tr=np.vstack((x_tr1,x_tr2))
  y_tr=fy(x_tr)+np.random.normal(0,.1,x_tr.shape)
  x_val=np.random.uniform(np.min(x_tr),np.max(x_tr),NS).reshape((-1,1))
  y_val=fy(x_val)
  return x_tr, y_tr, x_val, y_val

def f3():
  import pandas as pd
  from datetime import datetime as dt
  temps_data = pd.read_csv('../jac_band/french_1d.csv', delimiter=";")
  y_tr = temps_data[['t']].values-273.15
  x_temp = temps_data[['date']].values
  x_temp1=list(map(lambda d: dt.strptime(str(d)[1:11],'%Y%m%d%H'),x_temp))
  x_tr=np.array(list(map(lambda d: (d-x_temp1[0]).total_seconds()/3600,x_temp1))).reshape((-1,1))
  return x_tr, y_tr

def f4():
  x_tr=np.random.uniform(-2,2,50).reshape((-1,1))
  y_tr=x_tr**2*(x_tr<0)+x_tr*(x_tr>0)+np.random.normal(0,0.1,x_tr.shape)
  return x_tr, y_tr


x_tr, y_tr=f3()
x1=np.linspace(np.min(x_tr),np.max(x_tr),NS).reshape((-1,1))
xs=np.vstack((x_tr,x1))
xs_argsort=xs.argsort(0)
y1_kgd=kgd(xs,x_tr,y_tr, plot=True, step_size=0.01, sleep_time=0.1)
sys.exit()

for seed in range(10):
  np.random.seed(seed)
  x_tr, y_tr, x_val, y_val=f2b()
  n_tr=x_tr.shape[0]
  #x1=np.linspace(np.min(x_tr),np.max(x_tr),NS).reshape((-1,1))
  #xs=np.vstack((x_tr,x1))
  #xs_argsort=xs.argsort(0)
  
  xs=np.vstack((x_tr,x_val))
  xs_argsort=xs.argsort(0)
  
  r2_max=0
  for lbda in np.geomspace(1,0.001, 50):
    for sigma in np.geomspace(.1,0.01,50):
      y1_krr=krr(xs,x_tr,y_tr,lbda,sigma)
      if r2(y_val,y1_krr[n_tr:,:])>r2_max:
        r2_max=r2(y_val,y1_krr[n_tr:,:])
        best_vals=[lbda,sigma,r2(y_tr, y1_krr[:n_tr,:]),r2_max]
  print(best_vals)
#  y1_krr=krr(xs,x_tr,y_tr,best_vals[0], best_vals[1])
#  fig,ax=plt.subplots(1,1,figsize=(20,6))
#  ax.plot(x_tr,y_tr,'ok')
#  ax.plot(x_val,y_val,'or')
#  ax.plot(xs[xs_argsort,0],y1_krr[xs_argsort,0])
#  ax.plot([np.min(x_tr),np.min(x_tr)+sigma],2*[1.1*(np.min(np.vstack((y_tr,y1_krr))))],'C2',lw=3)
#  fig.savefig('figures/krr.pdf')
  
  
  y1_kgd=kgd(xs,x_tr,y_tr, plot=False, step_size=0.01, sleep_time=0.1,val_data=[x_val, y_val])
  print(r2(y_tr,y1_kgd[:n_tr,:]),r2(y_val,y1_kgd[n_tr:,:]))
  print('')

  y1_krr=krr(xs,x_tr,y_tr,best_vals[0], best_vals[1])
  fig,ax=plt.subplots(1,1,figsize=(20,6))
  ax.plot(x_tr,y_tr,'ok')
  ax.plot(x_val,y_val,'or')
  ax.plot(xs[xs_argsort,0],y1_krr[xs_argsort,0],'C0')
  ax.plot(xs[xs_argsort,0],y1_kgd[xs_argsort,0],'C2')
  fig.savefig('figures/krr.pdf')
