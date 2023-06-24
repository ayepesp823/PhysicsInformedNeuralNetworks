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
lr = 0.125
layers = np.array([2,30,30,30,30,30,30,30,30,1])#50,50,30,50,50
xmin, xmax = 0,1
ymin, ymax = 0,1
total_points_x = 100
total_points_y = 100
Nu, Nf = 500, 10000#? #Nu: number of training points of boundary #Nf: number of collocation points (to evaluate PDE in)

k0 = 2*np.pi*2
def f_real(x,t):
    return torch.sin(k0 * x) * torch.sin(k0 * t)
'Loss Function'
def ResPDE(Xpde,Fcomp):
    f_x_y = autograd.grad(Fcomp,Xpde,torch.ones([Xpde.shape[0], 1]).to(model.device), retain_graph=True, create_graph=True)[0] #first derivative
    f_xx_yy = autograd.grad(f_x_y,Xpde,torch.ones(Xpde.shape).to(model.device), create_graph=True)[0] #second derivative
    
    f_xx=f_xx_yy[:,[0]]# we select the 1st element for x (the second one is t) (Remember the input X=[x,t]) 
    f_yy=f_xx_yy[:,[1]]# we select the 1st element for x (the second one is t) (Remember the input X=[x,t]) 
    f=-f_yy - f_xx - (k0**2)*Fcomp - (k0**2)*torch.sin(k0 * Xpde[:, 0:1]) * torch.sin(k0 * Xpde[:, 1:2])
    return [f]

def transform(x, u):
    n=50;h=n/500
    res = torch.sigmoid(n*(x[:,[0]]-h))*torch.sigmoid(n*(1-x[:,[0]]-h))*torch.sigmoid(n*(x[:,[1]]-h))*torch.sigmoid(n*(1-x[:,[1]]-h))
    return res * u

'Model'

argsoptim = {'max_iter':30,'max_eval':None,'tolerance_grad':1e-11,
            'tolerance_change':1e-11,'history_size':100,'line_search_fn':'strong_wolfe'}

model = FCN(layers,ResPDE,transform,activation=torch.sin,lr=lr,optimizer='LBFGS',argsoptim=argsoptim,numequ=1,numcdt=4,scheduler='StepLR',argschedu={'gamma':2,'step_size':250},name='Helmholtz2DHC')#,30000,50000]})

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

def FCDT(Xcdt,Ucdt):
    uborder = torch.zeros(Xcdt.shape[0],1)
    return model.criterion(Ucdt,uborder)

CDT = [(left_X,FCDT),\
        (right_X,FCDT),\
        (bottom_X,FCDT),\
        (top_X,FCDT)]

# Collocation Points (Evaluate our PDe)
#Choose(Nf) points(Latin hypercube)
X_train_Nf=lb+(ub-lb)*lhs(2,Nf) # 2 as the inputs are x and t
Xcdt=torch.vstack([left_X,bottom_X,top_X,right_X])
X_train_Nf=torch.vstack((X_train_Nf,Xcdt)).float().to(model.device) #Add the training poinst to the collocation points
X_test=x_test.float().to(model.device) # the input dataset (complete)
Y_test=u_test.float().to(model.device)

fig, ax = model.Train(X_train_Nf,CDT=CDT,Verbose=True,nEpochs=steps,nLoss=100,Test=[X_test,Y_test],pScheduler=True,SH=True)
plt.close('all')
model.save()
model.load_state_dict(torch.load(f'Models/{model.name}/Best/StateDict_{model.n}.model'))

u_pred = model(x_test).detach().numpy()
x_test = x_test.detach().numpy()
x_test,y_test = x_test.T
u_test = u_test.detach().numpy()

norm1,mapper1 = ColorMapMaker(u_test.min(),u_test.max(),'cool')

fig, (axr,axp,axe) = plt.subplots(3,figsize=(16,25),sharex=True,sharey=True)

for i,ax in enumerate((axr,axp,axe)):
    # ax.grid(ls='none',axis='y',which='minor')
    ax.tick_params(axis='both',which='major',labelsize=20)

axr.scatter(x_test,y_test,c=mapper1.to_rgba(u_test),edgecolors='none',marker='s',s=55)
axp.scatter(x_test,y_test,c=mapper1.to_rgba(u_pred),edgecolors='none',marker='s',s=55)
err = u_pred-u_test

maxe = 0
if np.abs(err.min())>np.abs(err.max()):maxe = np.abs(err.min())
else:maxe = np.abs(err.max())

norm2,mapper2 = ColorMapMaker(-maxe,maxe,'seismic')

axe.scatter(x_test,y_test,c=mapper2.to_rgba(err),edgecolors='none',marker='s',s=55)

fig.subplots_adjust(hspace=0.04,wspace=0.025)
axr.set_xlim(xmin,xmax);axr.set_ylim(ymin,ymax)

cbar_col = plt.colorbar(mapper1,ax=[axr,axp],shrink=0.95,extend='both')
cbar_col.ax.tick_params(labelsize=15)
cbar_col.ax.set_ylabel(r'$u$ [a.u]', rotation=-90, labelpad = 10, fontdict = {"size":25})

cbar_err = plt.colorbar(mapper2,ax=[axe    ],shrink=0.95,extend='both')
cbar_err.ax.tick_params(labelsize=15)
cbar_err.ax.set_ylabel(r'$u_{err}$ [a.u]', rotation=-90, labelpad = 10, fontdict = {"size":25})

axr.text(0.5,0.8*ymax,r'Real Solution',horizontalalignment='center', verticalalignment='center',\
            color='k',fontsize=27)
axp.text(0.5,0.8*ymax,r'Predicted Solution',horizontalalignment='center', verticalalignment='center',\
            color='k',fontsize=27)
axp.set_ylabel(r'$y$ [a.u]',fontsize=25)
axe.text(0.5,0.8*ymax,r'Computed Error',horizontalalignment='center', verticalalignment='center',\
            color='k',fontsize=27)
axe.set_xlabel(r'$x$ [a.u]',fontsize=25)


fig.savefig(f'Figures/{model.name}/Images/Results.pdf',bbox_inches='tight')
fig.savefig(f'Figures/{model.name}/Images/Results.png',bbox_inches='tight')
