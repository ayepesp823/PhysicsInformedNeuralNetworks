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
steps = 25000
lr = 0.001
layers = np.array([2,32,32,1])#50,50,20,50,50
xmin, xmax = -1,1
tmin, tmax = 0,1
total_points_x = 200
total_points_t = 100
Nu, Nf = 100, 10000#? #Nu: number of training points of boundary #Nf: number of collocation points (to evaluate PDE in)

k0 = 2*np.pi*2
def f_real(x,t):
    return torch.exp(-t)*(torch.sin(np.pi*x))#torch.exp(-t)*(torch.sin(x)+(torch.sin(2*x))/2+(torch.sin(3*x))/3+(torch.sin(4*x))/4+(torch.sin(8*x))/8)#
'Loss Function'
def ResPDE(Xpde,Fcomp):
    f_x_t = autograd.grad(Fcomp,Xpde,torch.ones([Xpde.shape[0], 1]).to(model.device), retain_graph=True, create_graph=True)[0] #first derivative
    f_xx_tt = autograd.grad(f_x_t,Xpde,torch.ones(Xpde.shape).to(model.device), create_graph=True)[0] #second derivative
    
    f_t = f_x_t[:,[1]]
    f_xx=f_xx_tt[:,[0]]# we select the 1st element for x (the second one is t) (Remember the input X=[x,t]) 
    f= f_t-f_xx-torch.exp(-Xpde[:,[1]])*(3*(torch.sin(2*Xpde[:,[0]]))/2+8*(torch.sin(3*Xpde[:,[0]]))/3+15*(torch.sin(4*Xpde[:,[0]]))/4+63*(torch.sin(8*Xpde[:,[0]]))/8)#+torch.exp(-Xpde[:,[1]])*(torch.sin(np.pi*Xpde[:,[0]])-(np.pi**2)*torch.sin(np.pi*Xpde[:,[0]]))#
    return [f]

'Model'
# model = FCN(layers,ResPDE,lr=lr,scheduler='StepLR',argschedu={'gamma':0.5,'step_size':12500},name='ReactionDiffusion2',numcdt=3)

'Gen Data'
x=torch.linspace(xmin,xmax,total_points_x).view(-1,1)
t=torch.linspace(tmin,tmax,total_points_t).view(-1,1)
# Create the mesh 
X,T=torch.meshgrid(x.squeeze(1),t.squeeze(1))
# Evaluate real function
y_real=f_real(X,T)
# plot3D(x,t,y_real)
# Transform the mesh into a 2-column vector
x_test=torch.hstack((X.transpose(1,0).flatten()[:,None],T.transpose(1,0).flatten()[:,None]))
y_test=y_real.transpose(1,0).flatten()[:,None] # Colum major Flatten (so we transpose it)
# Domain bounds
lb=x_test[0] #first value
ub=x_test[-1] #last value 

#Initial Condition
#Left Edge: x(x,0)=sin(x)->xmin=<x=<xmax; t=0
left_X=torch.hstack((X[:,0][:,None],T[:,0][:,None])) # First column # The [:,None] is to give it the right dimension
left_Y=torch.sin(np.pi*left_X[:,0]).unsqueeze(1)#(torch.sin(left_X[:,0])+torch.sin(2*left_X[:,0])/2+torch.sin(3*left_X[:,0])/3+torch.sin(4*left_X[:,0])/4+torch.sin(8*left_X[:,0])/8).unsqueeze(1)
#Boundary Conditions
#Bottom Edge: x=min; tmin=<t=<max
bottom_X=torch.hstack((X[0,:][:,None],T[0,:][:,None])) # First row # The [:,None] is to give it the right dimension
bottom_Y=torch.zeros(bottom_X.shape[0],1)
#Top Edge: x=max; 0=<t=<1
top_X=torch.hstack((X[-1,:][:,None],T[-1,:][:,None])) # Last row # The [:,None] is to give it the right dimension
top_Y=torch.zeros(top_X.shape[0],1)
#Get all the training data into the same dataset
X_train=torch.vstack([left_X,bottom_X,top_X])
Y_train=torch.vstack([left_Y,bottom_Y,top_Y])

def FunCDT(Xcdt,Ycdt):
    Y_real = torch.zeros(Xcdt.shape[0],1)
    return model.criterion(Ycdt,Y_real)

def Fun1CDT(Xcdt,Ycdt):
    return model.criterion(Ycdt,left_Y)

model = torch.load('Models/ReactionDiffusion2/Final/Trained_2.model')

#Choose(Nu) points of our available training data:
idx=np.random.choice(X_train.shape[0],Nu,replace=False)
X_train_Nu=X_train[idx,:].float().to(model.device)
Y_train_Nu=Y_train[idx,:].float().to(model.device)
# Collocation Points (Evaluate our PDe)
#Choose(Nf) points(Latin hypercube)
X_train_Nf=lb+(ub-lb)*lhs(2,Nf) # 2 as the inputs are x and t
X_train_Nf=torch.vstack((X_train_Nf,X_train_Nu)).float().to(model.device) #Add the training poinst to the collocation points
X_test=x_test.float().to(model.device) # the input dataset (complete)
Y_test=y_test.float().to(model.device)

# fig, ax, fig2, ax2 = model.Train(X_train_Nf,CDT=[(left_X,Fun1CDT),(top_X,FunCDT),(bottom_X,FunCDT)],Verbose=True,nEpochs=steps,nLoss=1000,Test=[X_test,Y_test],pScheduler=True,SH=True)
# plt.close('all')
# model.save()
model.load_state_dict(torch.load(f'Models/{model.name}/Best/StateDict_{model.n}.model'))

y_pred = model(x_test).detach().numpy()
x_test = x_test.detach().numpy()
x_test,t_test = x_test.T
y_test = y_test.detach().numpy()
norm1,mapper1 = ColorMapMaker(y_test.min(),y_test.max(),'cool')

fig, (axr,axp,axe) = plt.subplots(3,figsize=(16,25),sharex=True,sharey=True)

for i,ax in enumerate((axr,axp,axe)):
    ax.grid(ls='none',axis='y',which='minor')
    ax.tick_params(axis='both',which='major',labelsize=20)

axr.scatter(t_test,x_test,c=mapper1.to_rgba(y_test),edgecolors='none',marker='s',s=65)
axp.scatter(t_test,x_test,c=mapper1.to_rgba(y_pred),edgecolors='none',marker='s',s=65)
err = 2*(y_test-y_pred)/(np.abs(y_test)+1)

maxe = 0
if np.abs(err.min())>np.abs(err.max()):maxe = np.abs(err.min())
else:maxe = np.abs(err.max())
# maxe *= 1.1

norm2,mapper2 = ColorMapMaker(-maxe,maxe,'seismic')

axe.scatter(t_test,x_test,c=mapper2.to_rgba(err),edgecolors='none',marker='s',s=65)

fig.subplots_adjust(hspace=0.04,wspace=0.025)
fig.subplots_adjust(right=0.81)
cbaru_ax = fig.add_axes([0.815, 0.38, 0.025, 0.49])
cbare_ax = fig.add_axes([0.815, 0.12, 0.025, 0.23])
axr.set_ylim(xmin,xmax);axr.set_xlim(tmin,tmax)

cbar_col = plt.colorbar(mapper1,cax=cbaru_ax,shrink=0.95,extend='both')
cbar_col.ax.tick_params(labelsize=15)
cbar_col.ax.set_ylabel(r'$y$ [a.u]', rotation=-90, labelpad = 15, fontdict = {"size":25})

cbar_err = plt.colorbar(mapper2,cax=cbare_ax,shrink=0.95,extend='both')
cbar_err.ax.tick_params(labelsize=15)
cbar_err.ax.set_ylabel(r'$y_{err}$ [a.u]', rotation=-90, labelpad = 15, fontdict = {"size":25})

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

