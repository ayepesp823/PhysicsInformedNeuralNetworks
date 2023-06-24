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

'Initial set up'
torch.manual_seed(1234)
np.random.seed(1234)
steps = 1000
lr = 1e-3
layers = np.array([2,80,80,50,80,80,1])#np.array([2,50,50,30,50,50,1])#50,50,30,50,50
xmin, xmax = -1,1
ymin, ymax = -1,1
total_points_x = 256
total_points_y = 256
Nu, Nf = 1000, 5000#? #Nu: number of training points of boundary #Nf: number of collocation points (to evaluate PDE in)

a_1 = 1
a_2 = 1

k = 1

def f_real(x,y):
    return torch.sin(a_1 * np.pi * x) * torch.sin(a_2 * np.pi * y)
'Loss Function'
def ResPDE(Xpde,Fcomp):
    f_x_y = autograd.grad(Fcomp,Xpde,torch.ones([Xpde.shape[0], 1]).to(model.device), retain_graph=True, create_graph=True)[0] #first derivative
    f_xx_yy = autograd.grad(f_x_y,Xpde,torch.ones(Xpde.shape).to(model.device), create_graph=True)[0] #second derivative
    
    f_xx=f_xx_yy[:,[0]]# we select the 1st element for x (the second one is t) (Remember the input X=[x,t]) 
    f_yy=f_xx_yy[:,[1]]# we select the 1st element for x (the second one is t) (Remember the input X=[x,t]) 
    
    f=f_yy + f_xx + k**2 * Fcomp - ( -(a_1*np.pi)**2 - (a_2*np.pi)**2 + k**2 ) * torch.sin(a_1*np.pi*Xpde[:,[0]]) * torch.sin(a_2*np.pi*Xpde[:,[1]])
    return [f]

'Model'

argsoptim = {'max_iter':50,'max_eval':None,'tolerance_grad':1e-35,
            'tolerance_change':1e-35,'history_size':1000,'line_search_fn':'strong_wolfe'}

model = FCN(layers,ResPDE,lr=lr,activation=torch.sin,numequ=1,numcdt=2,name='Helmholtz2D',optimizer='LBFGS',argsoptim=argsoptim)#,scheduler='StepLR',argschedu={'gamma':1.1,'step_size':250})#,30000,50000]})
model.focusmin=None;model.n=19

'Gen Data'
x=torch.linspace(xmin,xmax,total_points_x).view(-1,1)
t=torch.linspace(ymin,ymax,total_points_y).view(-1,1)
# Create the mesh 
X,Y=torch.meshgrid(x.squeeze(1),t.squeeze(1))
# Evaluate real function
u_real=f_real(X,Y)

# Transform the mesh into a 2-column vector
x_test=torch.hstack((X.transpose(1,0).flatten()[:,None],Y.transpose(1,0).flatten()[:,None]))
u_test=u_real.transpose(1,0).flatten()[:,None] # Colum major Flatten (so we transpose it)
# Domain bounds
lb=x_test[0] #first value
ub=x_test[-1] #last value 

#Boundary Conditions
#Left Edge: xmin=<x=<xmax; y=ymax
left_X=torch.hstack((X[:,0][:,None],Y[:,0][:,None])) # First column # The [:,None] is to give it the right dimension

#right Edge: xmin=<x=<xmax; y=ymin
right_X=torch.hstack((X[:,-1][:,None],Y[:,-1][:,None])) # First column # The [:,None] is to give it the right dimension

#Bottom Edge: x=xmin; ymin=<y=<ymax
bottom_X=torch.hstack((X[0,:][:,None],Y[0,:][:,None])) # First row # The [:,None] is to give it the right dimension

#Top Edge: x=xmax; ymin=<y=<ymax
top_X=torch.hstack((X[-1,:][:,None],Y[-1,:][:,None])) # Last row # The [:,None] is to give it the right dimension

cdt_X = torch.vstack((left_X,right_X,bottom_X,top_X))

idx = np.random.choice(x_test.shape[0],Nu,replace=False)
real_X = x_test[idx]

def FCDT(Xcdt,Ucdt):
    uborder = torch.zeros(Xcdt.shape[0],1)
    return model.criterion(Ucdt,uborder)

def FReal(Xcdt,Ucdt):
    real_U = f_real(Xcdt[:,[0]],Xcdt[:,[1]])
    return model.criterion(Ucdt,real_U)

CDT = [(cdt_X,FCDT,5),\
        (real_X,FReal,5)]


# Collocation Points (Evaluate our PDe)
#Choose(Nf) points(Latin hypercube)
X_train_Nf=lb+(ub-lb)*lhs(2,Nf) # 2 as the inputs are x and t
Xcdt = torch.vstack((left_X,right_X,bottom_X,top_X))
X_train_small = (-0.25,-0.25)+(0.5,0.5)*lhs(2,Nf)
X_train_small = torch.tensor(X_train_small)
X_train_Nf=torch.vstack((X_train_Nf,Xcdt,real_X,X_train_small)).float().to(model.device) #Add the training poinst to the collocation points
X_test=x_test.float().to(model.device) # the input dataset (complete)
Y_test=u_test.float().to(model.device)

# fig, ax, fig2, ax2 = model.Train(X_train_Nf,CDT=CDT,Verbose=True,nEpochs=steps,nLoss=50,Test=[X_test,Y_test],pScheduler=True,SH=True)
# plt.close('all')
# model.save()
model.load_state_dict(torch.load(f'Models/{model.name}/Best/StateDict_{model.n}.model'))

u_pred = model(x_test).detach().numpy()
x_test = x_test.detach().numpy()
x_test,y_test = x_test.T
u_test = u_test.detach().numpy()

norm1,mapper1 = ColorMapMaker(u_test.min(),u_test.max(),'cool')

fig, (axr,axp,axe) = plt.subplots(3,figsize=(16,25),sharex=True,sharey=True)

for i,ax in enumerate((axr,axp,axe)):
    ax.grid(ls='none',axis='y',which='minor')
    ax.tick_params(axis='both',which='major',labelsize=20)

axr.scatter(x_test,y_test,c=mapper1.to_rgba(u_test),edgecolors='none',marker='s',s=55)
axp.scatter(x_test,y_test,c=mapper1.to_rgba(u_pred),edgecolors='none',marker='s',s=55)
err = 2*(u_test-u_pred)/(np.abs(u_test)+1)

maxe = 0
if np.abs(err.min())>np.abs(err.max()):maxe = np.abs(err.min())
else:maxe = np.abs(err.max())

norm2,mapper2 = ColorMapMaker(-maxe,maxe,'seismic')

axe.scatter(x_test,y_test,c=mapper2.to_rgba(err),edgecolors='none',marker='s',s=55)

fig.subplots_adjust(hspace=0.04,wspace=0.025)
fig.subplots_adjust(right=0.81)
cbaru_ax = fig.add_axes([0.815, 0.38, 0.025, 0.49])
cbare_ax = fig.add_axes([0.815, 0.12, 0.025, 0.23])
axr.set_ylim(ymin,ymax);axr.set_xlim(xmin,xmax)

cbar_col = plt.colorbar(mapper1,cax=cbaru_ax,shrink=0.95,extend='both')
cbar_col.ax.tick_params(labelsize=15)
cbar_col.ax.set_ylabel(r'$u$ [a.u]', rotation=-90, labelpad = 15, fontdict = {"size":25})

cbar_err = plt.colorbar(mapper2,cax=cbare_ax,shrink=0.95,extend='both')
cbar_err.ax.tick_params(labelsize=15)
cbar_err.ax.set_ylabel(r'$u_{err}$ [a.u]', rotation=-90, labelpad = 15, fontdict = {"size":25})

axr.text(0,0.8*xmax,r'Analytical Solution',horizontalalignment='center', verticalalignment='center',\
            color='k',fontsize=27)
axp.text(0,0.8*xmax,r'Predicted Solution',horizontalalignment='center', verticalalignment='center',\
            color='k',fontsize=27)
axp.set_ylabel(r'$x$ [a.u]',fontsize=25)
axe.text(0,0.8*xmax,r'Computed Error',horizontalalignment='center', verticalalignment='center',\
            color='k',fontsize=27)
axe.set_xlabel(r'Time (t) [a.u]',fontsize=25)


fig.savefig(f'Figures/{model.name}/Images/Results{model.n}.pdf',bbox_inches='tight')
fig.savefig(f'Figures/{model.name}/Images/Results{model.n}.png',bbox_inches='tight')
