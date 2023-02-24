import numpy as np
from matplotlib import pyplot as plt
from time import sleep

def kgd(xs,x_tr,y_tr_in,sigma0=None,step_size=0.01,kern=None, sigma_min=0, t_max=1e4, plot=False, sleep_time=0.1, val_data=None):
  if plot:
    fig,ax=plt.subplots(1,1,figsize=(20,6))
    xs_argsort=xs.argsort(0)
  
  if kern is None:
    def kern(X,Y,sigma):
      import numpy as np
      if X.shape[1]==1 and Y.shape[1]==1:
        return np.exp(-0.5*((X-Y.T)/sigma)**2)
      X2=np.sum(X**2,1).reshape((-1,1))
      XY=X.dot(Y.T)
      Y2=np.sum(Y**2,1).reshape((-1,1))
      D2=X2-2*XY+Y2.T
      return np.exp(-0.5*D2/sigma**2)

  n_tr=x_tr.shape[0]
  ns=xs.shape[0]
  Ih=np.hstack((np.eye(n_tr),np.zeros((n_tr,ns-n_tr))))
  y_tr_mean=np.mean(y_tr_in)
  y_tr=y_tr_in-y_tr_mean
  
  sigma=1*(np.max(x_tr)-np.min(x_tr)) if sigma0 is None else sigma0
  Ks=kern(xs,x_tr,sigma)
  ys=np.zeros(xs.shape)
  r2=0
  for i in range(int(t_max/step_size)):
    ys-= step_size*Ks@(Ih@ys-y_tr)
    r2_old=r2
    r2=(1-np.mean((y_tr-ys[:n_tr,:])**2)/np.mean(y_tr**2))
    if (r2-r2_old)/step_size<0.05 and sigma>sigma_min:
      sigma=sigma/(1+step_size)
      Ks=kern(xs,x_tr,sigma)
    if r2>0.9999 or (plot and i%100/step_size==0):
      if plot:
        print(sigma)
        ax.cla()
        ax.plot(x_tr,y_tr+y_tr_mean,'ok')
        if not val_data is None:
          ax.plot(val_data[0], val_data[1], 'or')
        ax.plot(xs[xs_argsort,0],ys[xs_argsort,0]+y_tr_mean)
        ax.plot([np.min(x_tr),np.min(x_tr)+sigma],2*[1.1*(np.min(np.vstack((y_tr,ys)))+y_tr_mean)],'C2',lw=3)
        fig.savefig('figures/kgd.pdf')
        sleep(sleep_time)
      if r2>0.9999:
        break
  return ys+y_tr_mean

