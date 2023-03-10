import numpy as np
from matplotlib import pyplot as plt
from time import sleep

def kgd3_pol(xs,x_tr,y_tr,step_size=0.01,p_max=np.inf, t_max=1e4, plot=False):
  if plot:
    fig,ax=plt.subplots(1,1,figsize=(12,12))
    xs_argsort=xs.argsort(0)
  kern = lambda x1, x2, p: (1+x1*x2.T)**p
  p=1
  y1=np.zeros(xs.shape)
  Ks=kern(xs,x_tr,p)
  gd_obj=gd_alg(Ks,y_tr,'gd','pred',step_size)
  n_tr=y_tr.shape[0]
  r2=0
  for i in range(1000):
    gd_obj.gd_step()
    y1=gd_obj.get_fs()
    r2_old=r2
    r2=(1-np.mean((y_tr-y1[:n_tr,:])**2)/np.mean((y_tr-np.mean(y_tr))**2))
    if (r2-r2_old)/step_size<0.01 and p<p_max:
      p+=1
      print(i,p)
      Ks=kern(xs,x_tr,p)
      gd_obj.update(Ks,y1)
      if plot:# and i%10/step_size==0:
        print(p)
        ax.cla()
        ax.plot(x_tr,y_tr,'ok')
        ax.plot(xs[xs_argsort,0],y1[xs_argsort,0])
        fig.savefig('figures/kgd3.pdf')
        sleep(1)
    if r2>0.999:
      break
  if plot:
    print(p)
    ax.cla()
    ax.plot(x_tr,y_tr,'ok')
    ax.plot(xs[xs_argsort,0],y1[xs_argsort,0])
    fig.savefig('figures/kgd3.pdf')
    sleep(.1)
  return gd_obj.get_fs()


def kgd3(xs,x_tr,y_tr,sigma0=None,step_size=0.01,kern=None, sigma_min=0, t_max=1e4, plot=False):
  if plot:
    fig,ax=plt.subplots(1,1,figsize=(12,12))
    xs_argsort=xs.argsort(0)
  sigma=np.max(x_tr)-np.min(x_tr) if sigma0 is None else sigma0
  if kern is None:
    kern = lambda x1, x2, sigma: np.exp(-0.5*np.square((x1-x2.T)/sigma))
  y1=np.zeros(xs.shape)
  Ks=kern(xs,x_tr,sigma)
  gd_obj=gd_alg(Ks,y_tr,'gd','pred',step_size)
  n_tr=y_tr.shape[0]
  r2=0
  for i in range(int(t_max/step_size)):
    gd_obj.gd_step()
    y1=gd_obj.get_fs()
    r2_old=r2
    r2=(1-np.mean((y_tr-y1[:n_tr,:])**2)/np.mean((y_tr-np.mean(y_tr))**2))
    if (r2-r2_old)/step_size<0.05 and sigma>sigma_min:
      sigma=sigma/(1+step_size)
      Ks=kern(xs,x_tr,sigma)
      gd_obj.update(Ks,y1)
    if r2>0.9999:
      break
    if plot and i%100/step_size==0:
      print(sigma)
      ax.cla()
      ax.plot(x_tr,y_tr,'ok')
      ax.plot(xs[xs_argsort,0],y1[xs_argsort,0])
      fig.savefig('figures/kgd3.pdf')
      sleep(1)
  if plot:
    print(sigma)
    ax.cla()
    ax.plot(x_tr,y_tr,'ok')
    ax.plot(xs[xs_argsort,0],y1[xs_argsort,0])
    fig.savefig('figures/kgd3.pdf')
    sleep(1)
  return gd_obj.get_fs()


def kgd2(xs,x_tr,y_tr,t,sigma0=None,step_size=0.01,kern=None, sigma_min=None):
  sigma=np.max(x_tr)-np.min(x_tr) if sigma0 is None else sigma0
  if kern is None:
    kern = lambda x1, x2, sigma: np.exp(-0.5*np.square((x1-x2.T)/sigma))
  y1=np.zeros(xs.shape)
  Ks=kern(xs,x_tr,sigma)
  gd_obj=gd_alg(Ks,y_tr,'gd','pred',step_size)
  n_tr=y_tr.shape[0]
  r2=0
  r2s=[]
  sigmas=[]
  for i in range(int(t/step_size)):
    gd_obj.gd_step()
    y1=gd_obj.get_fs()
    r2_old=r2
    r2=(1-np.mean((y_tr-y1[:n_tr,:])**2)/np.mean((y_tr-np.mean(y_tr))**2))
    if (r2-r2_old)/step_size<0.01:
    #if True:
      #sigma=sigma/(1+alpha*step_size*sigma)
      sigma=sigma/(1+step_size)
      Ks=kern(xs,x_tr,sigma)
      gd_obj.update(Ks,y1)
    r2s.append(r2)
    sigmas.append(sigma)
    if r2>0.999:
      break
  return gd_obj.get_fs(), i*step_size, r2s, sigmas


def kgd1(xs,x_tr,y_tr,y_val,sigmas,step_size=0.01):
  kern = lambda x1, x2, sigma: np.exp(-0.5*np.square((x1-x2.T)/sigma))
  Ks=kern(xs,x_tr,sigmas[0])
  gd_obj=gd_alg(Ks,y_tr,'gd','pred',step_size)
  n_tr=y_tr.shape[0]
  r2s=[]
  for sigma in sigmas:
    gd_obj.gd_step()
    y1=gd_obj.get_fs()
    r2s.append(1-np.mean((y_val-y1[n_tr:,:])**2)/np.mean((y_val-np.mean(y_val))**2))
    Ks=kern(xs,x_tr,sigma)
    gd_obj.update(Ks,y1)
  return r2s

def kgd(xs,x_tr,y_tr,t,sigma0=None,step_size=0.01,gamma=0,alpha=0,kern=None, sigma_min=None):
  sigma=np.max(x_tr)-np.min(x_tr) if sigma0 is None else sigma0
  if kern is None:
    kern = lambda x1, x2, sigma: np.exp(-0.5*np.square((x1-x2.T)/sigma))
  y1=np.zeros(xs.shape)
  Ks=kern(xs,x_tr,sigma)
  gd_obj=gd_alg(Ks,y_tr,'gd','pred',step_size, gamma=gamma)
  for i in range(int(t/step_size)):
    y1_old=np.copy(y1)
    gd_obj.gd_step()
    y1=gd_obj.get_fs()
    if np.max(np.abs(y1))>1000 or np.max(np.abs(y1-y1_old))<1e-5:
      break
    #Update kernel
    if alpha>0:
      if sigma_min is None:
        sigma=sigma/(1+alpha*step_size*sigma)
      else:
        sigma=(sigma-alpha*step_size*sigma_min**2)/(1+alpha*step_size*(sigma-2*sigma_min))
      Ks=kern(xs,x_tr,sigma)
      gd_obj.update(Ks,y1)
  return gd_obj.get_fs(), i*step_size

class gd_alg():
  def __init__(self, K, y, alg, space='par', lr=0.01, gamma=0, alpha_egd=0.5, var0=None):
    assert space in ['par', 'pred', 'orac'], 'Non-valid space!'
    assert alg in ['cd', 'gd', 'egd', 'sgd', 'esgd', 'adam'], 'Non-valid alg!'
    self.K=K
    self.y=y
    self.alg=alg
    self.space=space
    self.lr=lr
    self.gamma=gamma
    self.alpha_egd=alpha_egd
    n=y.shape[0]
    if space=='par':
      if var0 is None:
        self.var=np.zeros((n,1))
      else:
        self.var=var0
      self.m=np.zeros((n,1))
      self.v=np.zeros((n,1))
    elif space=='pred' or space=='orac':
      ns=K.shape[0]
      if var0 is None:
        self.var=np.zeros((ns,1))
      else:
        self.var=var0
      self.Ih=np.hstack((np.eye(n),np.zeros((n,ns-n))))
      self.m=np.zeros((ns,1))
      self.v=np.zeros((ns,1))
    self.var_old=np.copy(self.var)

    self.b1=0.9
    self.b2=0.999
    self.eps=1e-7
    self.t=1

  def update(self,K,var):
    self.K=K
    self.var=var

  def gd_step(self):
    if self.space=='par':
      grad = self.K@self.var-self.y 
    elif self.space=='pred':
      grad = self.K@(self.Ih@self.var-self.y)
    elif self.space=='orac':
      grad = self.K@(self.var-self.y)
    if self.alg=='cd':
      I_cd=(np.abs(grad)==np.max(np.abs(grad)))
      self.var-= self.lr*I_cd*np.sign(grad)
    elif self.alg=='gd':
      self.var-=self.lr*grad
      #diff=self.var-self.var_old
      #self.var_old=np.copy(self.var)
      #self.var+=self.gamma*diff-self.lr*grad
    elif self.alg=='egd':
      I_egd=(np.abs(grad)>=self.alpha_egd*np.max(np.abs(grad)))
      #self.var-= self.lr*I_egd*grad
      self.var-= self.lr*I_egd*(self.alpha_egd*np.sign(grad)+(1-self.alpha_egd)*grad)
    elif self.alg=='sgd':
      self.var-=self.lr*np.sign(grad)
      #self.var-=self.lr*np.sign(np.sign(grad)*np.maximum((np.abs(grad)-0.001),0))
    elif self.alg=='sgd2':
      self.var-=self.lr*grad/((1-self.alpha_egd)*np.abs(grad)+self.alpha_egd)
      #self.var-=self.lr*np.sign(np.sign(grad)*np.maximum((np.abs(grad)-0.001),0))
    elif self.alg=='esgd':
      self.var-= self.lr*(self.alpha_egd*np.sign(grad)+(1-self.alpha_egd)*grad)
    elif self.alg=='adam':
      self.m=self.b1*self.m+(1-self.b1)*grad
      self.v=self.b2*self.v+(1-self.b2)*grad**2
      mh=self.m/(1-self.b1**self.t)
      vh=self.v/(1-self.b2**self.t)
      self.var-=self.lr/(np.sqrt(vh)+self.eps)*mh
      self.t+=1
    else:
      print('Non-valid algorithm!')
  
  def get_var(self):
    return self.var

  def get_fs(self):
    return self.get_var()

  def get_alpha(self):
    return self.get_var()

  def get_grad(self):
    if self.space=='par':
      grad = self.K@self.var-self.y 
    elif self.space=='pred':
      grad = self.K@(self.Ih@self.var-self.y)
    return grad

class prox_grad():
  def __init__(self, K,y,lbda, nrm, space='par', lr=0.001):
    assert space in ['par', 'pred'], 'Non-valid space!'
    assert nrm in ['l1','l2','linf'], "Non-valid norm!"
    self.K=K
    self.y=y
    self.lbda=lbda
    self.nrm=nrm
    self.space=space
    self.lr=lr
    n=y.shape[0]
    if space=='par':
      self.var=np.zeros((n,1))
    elif space=='pred':
      ns=K.shape[0]
      self.var=np.zeros((ns,1))
      self.Ih=np.hstack((np.eye(n),np.zeros((n,ns-n))))

  def prox(self, x):
    if self.nrm=='l1':
      return np.sign(x)*np.maximum(np.abs(x)-self.lbda,0)
    if self.nrm=='l2':
      return x/(1+self.lbda)
    if self.nrm=='linf':
      return x-self.euclidean_proj_l1ball(np.squeeze(x),self.lbda).reshape((-1,1))

  def prox_step(self):
    grad=self.K@self.var-self.y if self.space=='par' else self.K@(self.Ih@self.var-self.y)
    var_g=self.var-self.lr*grad
    self.var=self.prox(var_g)
  
  def get_var(self):
    return self.var

  def get_fs(self):
    return self.get_var()

  def get_alpha(self):
    return self.get_var()

  def euclidean_proj_simplex(self, v, s=1):
      """ Compute the Euclidean projection on a positive simplex
      Solves the optimisation problem (using the algorithm from [1]):
          min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
      Parameters
      ----------
      v: (n,) numpy array,
         n-dimensional vector to project
      s: int, optional, default: 1,
         radius of the simplex
      Returns
      -------
      w: (n,) numpy array,
         Euclidean projection of v on the simplex
      Notes
      -----
      The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
      Better alternatives exist for high-dimensional sparse vectors (cf. [1])
      However, this implementation still easily scales to millions of dimensions.
      References
      ----------
      [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
          John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
          International Conference on Machine Learning (ICML 2008)
          http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
      """
      assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
      n, = v.shape  # will raise ValueError if v is not 1-D
      # check if we are already on the simplex
      if v.sum() == s and np.alltrue(v >= 0):
          # best projection: itself!
          return v
      # get the array of cumulative sums of a sorted (decreasing) copy of v
      u = np.sort(v)[::-1]
      cssv = np.cumsum(u)
      # get the number of > 0 components of the optimal solution
      rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
      # compute the Lagrange multiplier associated to the simplex constraint
      theta = float(cssv[rho] - s) / (rho+1)
      # compute the projection by thresholding v using theta
      w = (v - theta).clip(min=0)
      return w
  
  
  def euclidean_proj_l1ball(self, v, s=1):
      """ Compute the Euclidean projection on a L1-ball
      Solves the optimisation problem (using the algorithm from [1]):
          min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
      Parameters
      ----------
      v: (n,) numpy array,
         n-dimensional vector to project
      s: int, optional, default: 1,
         radius of the L1-ball
      Returns
      -------
      w: (n,) numpy array,
         Euclidean projection of v on the L1-ball of radius s
      Notes
      -----
      Solves the problem by a reduction to the positive simplex case
      See also
      --------
      euclidean_proj_simplex
      """
      assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
      n, = v.shape  # will raise ValueError if v is not 1-D
      # compute the vector of absolute values
      u = np.abs(v)
      # check if v is already a solution
      if u.sum() <= s:
          # L1-norm is <= s
          return v
      # v is not already a solution: optimum lies on the boundary (norm == s)
      # project *u* on the simplex
      w = self.euclidean_proj_simplex(u, s=s)
      # compute the solution to the original problem on v
      w *= np.sign(v)
      return w
  
