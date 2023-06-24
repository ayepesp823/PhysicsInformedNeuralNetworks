import torch
from torch import  autograd
from torch import nn

from matplotlib import pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker
from sklearn.model_selection import train_test_split

import numpy as np
from pyDOE import lhs         #Latin Hypercube Sampling
from scipy import io
from scipy.stats import median_abs_deviation as MAD
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from time import time
from os import listdir
import warnings
from os.path import exists
from os import makedirs, listdir, remove
from matplotlib import cm
import matplotlib.colors as mcol
import matplotlib
warnings.filterwarnings("ignore")
try:plt.style.use('../figstyle.mplstyle')
except:pass

from PINNPDE import FCN

def ColorMapMaker(VMin,VMax,map):
    norm = matplotlib.colors.Normalize(vmin=VMin, vmax=VMax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=map)
    return norm, mapper


'Complementary function'
def plot3D(x,t,y):
    x_plot =x.squeeze(1) 
    t_plot =t.squeeze(1)
    X,T= torch.meshgrid(x_plot,t_plot)
    F_xt = y
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(T,X, F_xt,20,cmap="rainbow")
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('F(x,t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(),cmap="rainbow")
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f(x,t)')
    plt.show()

def plot3D_Matrix(x,t,y):
    X,T= x,t
    F_xt = y
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(T,X, F_xt,20,cmap="rainbow")
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('F(x,t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(),cmap="rainbow")
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f(x,t)')
    plt.show()

def solutionplot(u_pred,X_u_train,u_train):
    #https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks
    
    fig, ax = plt.subplots()
    ax.axis('off')

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(u_pred.detach().numpy(), interpolation='nearest', cmap='rainbow', 
                extent=[T.min(), T.max(), X.min(), X.max()], 
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(x,t)$', fontsize = 10)
    
    ''' 
    Slices of the solution at points t = 0.25, t = 0.50 and t = 0.75
    '''
    
    ####### Row 1: u(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,usol.T[25,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,u_pred.T[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')    
    ax.set_title('$t = 0.25s$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,usol.T[50,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,u_pred.T[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.50s$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,usol.T[75,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,u_pred.T[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])    
    ax.set_title('$t = 0.75s$', fontsize = 10)
    
    plt.savefig('Burgers.png',dpi = 500)   


'Initial set up'
torch.manual_seed(1234)
np.random.seed(1234)
steps = 1000
lr = 0.1
layers = np.array([2,20,20,20,20,20,20,20,2])
nu = 0.01/np.pi #diffusion coeficient

Nu, Nf = 100, 10000#? #Nu: number of training points of boundary #Nf: number of collocation points (to evaluate PDE in)


# def f_real(x,t):
#     return torch.exp(-t)*(torch.sin(x)+(torch.sin(2*x))/2+(torch.sin(3*x))/3+(torch.sin(4*x))/4+(torch.sin(8*x))/8)

'Loss Function'
def ResPDE(Xpde,Fcomp):
    Fcomp, y = Fcomp.T
    Fcomp = Fcomp.view(-1,1)
    f_x_t = autograd.grad(Fcomp,Xpde,torch.ones([Xpde.shape[0], 1]).to(model.device), retain_graph=True, create_graph=True)[0] #first derivative
    f_xx_tt = autograd.grad(f_x_t,Xpde,torch.ones(Xpde.shape).to(model.device), create_graph=True)[0] #second derivative
    
    f_x = f_x_t[:,[0]]
    f_t = f_x_t[:,[1]]

    f_xx=f_xx_tt[:,[0]]# we select the 1st element for x (the second one is t) (Remember the input X=[x,t]) 
    f= f_t + Fcomp*f_x - nu*f_xx
    return [f]

'Model'
argsoptim = {'max_iter':20,'max_eval':None,'tolerance_grad':1e-11,
            'tolerance_change':1e-11,'history_size':100,'line_search_fn':'strong_wolfe'}

model = FCN(layers,ResPDE,lr=lr,optimizer='LBFGS',argsoptim=argsoptim,name='Burger2D',numcdt=1)#,scheduler='StepLR',argschedu={'gamma':0.3,'step_size':3000})

'Gen Data'

data = io.loadmat('Data/burgers_shock.mat') 
x = data['x']   
xmin,xmax=x.min(),x.max()                  
t = data['t']                    
tmin,tmax=t.min(),t.max()               
usol = data['usol']           

X, T = np.meshgrid(x,t)

X_test = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))

# Domain bounds
lb = X_test[0]  # [-1. 0.]
ub = X_test[-1] # [1.  0.99]
u_true = usol.flatten('F')[:,None]

'''Boundary Conditions'''

#Initial Condition -1 =< x =<1 and t = 0  
left_X = np.hstack((X[0,:][:,None], T[0,:][:,None])) #L1
left_U = usol[:,0][:,None]

#Boundary Condition x = -1 and 0 =< t =<1
bottom_X = np.hstack((X[:,0][:,None], T[:,0][:,None])) #L2
bottom_U = usol[-1,:][:,None]

#Boundary Condition x = 1 and 0 =< t =<1
top_X = np.hstack((X[:,-1][:,None], T[:,0][:,None])) #L3
top_U = usol[0,:][:,None]

X_train = np.vstack([left_X, bottom_X, top_X]) 
U_train = np.vstack([left_U, bottom_U, top_U]) 

#choose random N_u points for training
idx = np.random.choice(X_train.shape[0], Nu, replace=False) 

X_train_Nu = X_train[idx, :] #choose indices from  set 'idx' (x,t)
U_train_Nu = U_train[idx,:]      #choose corresponding u

'''Collocation Points'''

# Latin Hypercube sampling for collocation points 
# N_f sets of tuples(x,t)
X_train_Nf = lb + (ub-lb)*lhs(2,Nf) 
X_train_Nf = np.vstack((X_train_Nf, X_train_Nu))

'Convert to tensor and send to GPU'
X_train_Nf = torch.from_numpy(X_train_Nf).float().to(model.device)
X_train_Nu = torch.from_numpy(X_train_Nu).float().to(model.device)
U_train_Nu = torch.from_numpy(U_train_Nu).float().to(model.device)
X_test = torch.from_numpy(X_test).float().to(model.device)
u = torch.from_numpy(u_true).float().to(model.device)
print(X_test.size(),u.size())

def FunCDT(Xcdt,Ycdt):
    return model.criterion(Ycdt,U_train_Nu)


fig, ax = model.Train(X_train_Nf,CDT=[(X_train_Nu,FunCDT,10)],Verbose=True,nEpochs=steps,nLoss=100,Test=[X_test,u],pScheduler=True)

name = 'Burger2D'
# n = len([file.replace(name,'').replace('.model','') for file in listdir('Models') if file.startswith(name)])
model.save()

# fig.savefig(f'Figures/Images/{name}Loss{n}.png',bbox_inches='tight')
plt.close(fig)
model.load_state_dict(torch.load(f'Models/{model.name}/Best/StateDict_{model.n}.model'))

u_pred,_ =model(X_test).T
u_pred = u_pred.view(-1,1)
# solutionplot(u_pred,X_train_Nu.cpu().detach().numpy(),U_train_Nu)

x1=X_test[:,0]
t1=X_test[:,1]
arr_x1=x1.reshape(shape=X.shape).transpose(1,0).detach().cpu().detach().numpy()
arr_T1=t1.reshape(shape=X.shape).transpose(1,0).detach().cpu().detach().numpy()
arr_y1=u_pred.reshape(shape=X.shape).transpose(1,0).detach().cpu().detach().numpy()
arr_y_test=usol


ax = plt.axes(projection='3d')
ax.plot_surface(arr_T1, arr_x1, arr_y1,cmap="rainbow")
ax.plot_surface(arr_T1, arr_x1, arr_y_test,color='k',alpha=0.8)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('f(x,t)')
fig = plt.gcf()

with open(f'Figures/{model.name}/Instances/Result_{model.n}.pkl','wb') as file:
    pkl.dump(fig,file)

plt.close()

norm1,mapper1 = ColorMapMaker(min(arr_y1.min(),arr_y_test.min()),max(arr_y1.max(),arr_y_test.max()),'cool')
common_levels = np.linspace(min(arr_y1.min(),arr_y_test.min()),max(arr_y1.max(),arr_y_test.max()),21)
fig, (axr,axp,axe) = plt.subplots(3,figsize=(16,25),sharex=True,sharey=True)

for i,ax in enumerate((axr,axp,axe)):
    ax.grid(ls='none',axis='y',which='minor')
    ax.tick_params(axis='both',which='major',labelsize=20)

axr.contourf(arr_T1, arr_x1, arr_y_test,common_levels,cmap='cool')
axp.contourf(arr_T1, arr_x1, arr_y1,common_levels,cmap='cool')
err = 2*(arr_y_test-arr_y1)/(1+np.abs(arr_y_test))

maxe = 0
if np.abs(err.min())>np.abs(err.max()):maxe = np.abs(err.min())
else:maxe = np.abs(err.max())
# maxe *= 1.2

norm2,mapper2 = ColorMapMaker(-maxe,maxe,'seismic')
levels2 = np.linspace(-maxe,maxe,21)


axe.contourf(arr_T1, arr_x1, err,levels2,cmap='seismic')

fig.subplots_adjust(hspace=0.04,wspace=0.025)
fig.subplots_adjust(right=0.81)
cbaru_ax = fig.add_axes([0.815, 0.38, 0.025, 0.49])
cbare_ax = fig.add_axes([0.815, 0.12, 0.025, 0.23])
axr.set_ylim(xmin,xmax);axr.set_xlim(tmin,tmax)

cbar_col = plt.colorbar(mapper1,cax=cbaru_ax,shrink=0.95,extend='both')
cbar_col.ax.tick_params(labelsize=15)
cbar_col.ax.set_ylabel(r'$u$ [a.u]', rotation=-90, labelpad = 15, fontdict = {"size":25})

cbar_err = plt.colorbar(mapper2,cax=cbare_ax,shrink=0.95,extend='both')
cbar_err.ax.tick_params(labelsize=15)
cbar_err.ax.set_ylabel(r'$u_{err}$ [a.u]', rotation=-90, labelpad = 15, fontdict = {"size":25})

axr.text(0.5,0.8*xmax,r'Analytical Solution',horizontalalignment='center', verticalalignment='center',\
            color='k',fontsize=27)
axp.text(0.5,0.8*xmax,r'Predicted Solution',horizontalalignment='center', verticalalignment='center',\
            color='k',fontsize=27)
axp.set_ylabel(r'$x$ [a.u]',fontsize=25)
axe.text(0.5,0.8*xmax,r'Computed Error',horizontalalignment='center', verticalalignment='center',\
            color='k',fontsize=27)
axe.set_xlabel(r'Time (t) [a.u]',fontsize=25)



fig.savefig(f'Figures/{model.name}/Images/Results.pdf',bbox_inches='tight')
fig.savefig(f'Figures/{model.name}/Images/Results.png',bbox_inches='tight')


# plot3D_Matrix(arr_x1,arr_T1,torch.from_numpy(arr_y1))