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
from scipy import special as sc
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
steps = 1000#0
lr = 1e-2
layers = np.array([3] + 7*[75] + [1])
xmin, xmax = 0,1
ymin, ymax = 0,1
tmin, tmax = 0,1
numx = 201
numy = 201
numt = 201
Nu, Nf = 750, 10000#? #Nu: number of training points of boundary #Nf: number of collocation points (to evaluate PDE in)
n,m,c,a = 3,3,1,1
omega = (c*np.pi/2)*np.sqrt((n**2)+(m**2))

def f_real(x,y,t):
    return a*torch.cos(omega*t)*torch.sin(n*np.pi*x)*torch.sin(m*np.pi*y)

def Make3DMesh(xmin,xmax,ymin,ymax,tmin,tmax,numx,numy,numt):
    x = torch.linspace(xmin,xmax,numx).view(-1,1)
    y = torch.linspace(ymin,ymax,numy).view(-1,1)
    t = torch.linspace(tmin,tmax,numt).view(-1,1)
    xxx, yyy, ttt = torch.meshgrid(x.squeeze(1),y.squeeze(1),t.squeeze(1))
    xxx = xxx.flatten()[:, None]
    yyy = yyy.flatten()[:, None]
    ttt = ttt.flatten()[:, None]
    Mesh = torch.hstack((xxx,yyy,ttt)).float().to(model.device)
    return Mesh

'Loss Function'
def ResPDE(Xpde,Ucomp):
    u_x_y_t = autograd.grad(Ucomp,Xpde,torch.ones([Xpde.shape[0], 1]).to(model.device), retain_graph=True, create_graph=True)[0] #first derivative
    u_xx_yy_tt = autograd.grad(u_x_y_t,Xpde,torch.ones(Xpde.shape).to(model.device), create_graph=True)[0] #second derivative
    
    u_xx=u_xx_yy_tt[:,[0]]# we select the 1st element for x (the second one is t) (Remember the input X=[x,t]) 
    u_yy=u_xx_yy_tt[:,[1]]# we select the 1st element for x (the second one is t) (Remember the input X=[x,t]) 
    u_tt=u_xx_yy_tt[:,[2]]# we select the 1st element for x (the second one is t) (Remember the input X=[x,t]) 
    
    f1=u_tt-(c**2)*(u_xx+u_yy)
    return [f1]

'Model'
argsoptim = {'max_iter':20,'max_eval':None,'tolerance_grad':1e-11,
            'tolerance_change':1e-11,'history_size':100,'line_search_fn':'strong_wolfe'}

model = FCN(layers,ResPDE,activation=torch.sin,lr=lr,numequ=1,numcdt=2,\
            name='RectangularMembrane',optimizer='LBFGS',argsoptim=argsoptim,\
            scheduler='StepLR',argschedu={'gamma':1.5,'step_size':100})
model.focusmin=None
model.n=3

#? Initial Conditions

#todo xyt for initial conditions [x,y,tmin]
ic_X = Make3DMesh(xmin,xmax,ymin,ymax,tmin,tmin,numx,numy,1)
idx = np.random.choice(ic_X.shape[0],2*Nu,replace=False)
ic_X = ic_X[idx,:]

#todo conditios at t=tmin=0; u=0,v=0
def InitialConditionCDT(Xcdt,Ycdt):
    Yreal = f_real(Xcdt[:,[0]],Xcdt[:,[1]],Xcdt[:,[2]])
    return model.criterion(Ycdt,Yreal)

#? Walls
#* Lower Wall
lower_wall = Make3DMesh(xmin,xmax,ymin,ymin,tmin,tmax,2*numx,1,numt)
#* Upper Wall
upper_wall = Make3DMesh(xmin,xmax,ymax,ymax,tmin,tmax,2*numx,1,numt)
#* Left Wall
left_wall = Make3DMesh(xmin,xmin,ymin,ymax,tmin,tmax,1,2*numy,numt)
#* Right Wall
right_wall = Make3DMesh(xmax,xmax,ymin,ymax,tmin,tmax,1,2*numy,numt)

wall_X = torch.vstack((lower_wall,upper_wall,left_wall,right_wall)).float().to(model.device)
idx = np.random.choice(wall_X.shape[0],3*Nu,replace=False)
wall_X = wall_X[idx,:]

#todo conditions at cylinder wall; u=0,v=0
def WallsCDT(Xcdt,Ycdt):
    wall = torch.zeros(Xcdt.shape[0],1).float().to(model.device)
    return model.criterion(Ycdt,wall)

#? Collocation points for equation residual

lb = torch.tensor([xmin,ymin,tmin])
ub = torch.tensor([xmax,ymax,tmax])

X_train_Nf = lb + (ub-lb)*lhs(3,Nf)
X_train_Nf = torch.vstack((X_train_Nf,wall_X)).float().to(model.device)

CDT = [(ic_X,InitialConditionCDT,3),\
        (wall_X,WallsCDT,4)]

X_test = Make3DMesh(xmin,xmax,ymin,ymax,tmin,tmax,numx,numy,numt)
idx = np.random.choice(X_test.shape[0],5*Nu,replace=False)
X_test = X_test[idx,:]
Y_test = f_real(X_test[:,0],X_test[:,1],X_test[:,2])

# fig, ax, fig2, ax2 = model.Train(X_train_Nf,CDT=CDT,Verbose=True,nEpochs=steps,nLoss=50,Test=[X_test,Y_test],pScheduler=True,SH=True,RE=True)
# plt.close('all')
# model.save()
model.load_state_dict(torch.load(f'Models/{model.name}/Best/StateDict_{model.n}.model'))

x_cyl = torch.linspace(xmin,xmax,101)
y_cyl = torch.linspace(ymin,ymax,101)
x_cyl, y_cyl = torch.meshgrid(x_cyl,y_cyl)
x_cyl = x_cyl.flatten()[:, None]
y_cyl = y_cyl.flatten()[:, None]

XY = torch.hstack((x_cyl,y_cyl))

Titles = ['Analytical','Predicted','Error']

def getxy(XY):
    x = XY[:,[0]].numpy().reshape(101,101)
    y = XY[:,[1]].numpy().reshape(101,101)
    return x,y

if not exists(f'Figures/{model.name}/Images/Evolution{model.n}'):
    makedirs(f'Figures/{model.name}/Images/Evolution{model.n}')

fig, axes = plt.subplots(3,5,figsize=(16,9))

axesr = axes[0,:]
axesp = axes[1,:]
axese = axes[2,:]

levelsu = np.linspace(-1.05,1.05,73)
levelse = np.linspace(-2.1,2.1,181)

normu, cmapu = ColorMapMaker(-1.05,1.05,'turbo')
norme, cmape = ColorMapMaker(-2.1,2.1,'seismic')

fig.subplots_adjust(hspace=0.035,wspace=0.02)
fig.subplots_adjust(right=0.83)

cbaru_ax = fig.add_axes([0.835, 0.38, 0.025, 0.49])
# cbaru_ax.set_xticks([])
# cbaru_ax.yaxis.tick_right()
# cbaru_ax.tick_params(axis='both',which='major',labelsize=15)
# cbaru_ax.set_ylabel(r'Displacement ($u$)',fontsize=18,rotation=-90)
# cbaru_ax.yaxis.set_label_position("right")
# cbaru_ax.set_ylim(-1.2,1.2)
cbaru = plt.colorbar(cmapu,cax=cbaru_ax,shrink=0.95,extend='both',format='%+.1f')
cbaru.ax.tick_params(labelsize=15)
cbaru.ax.set_ylabel(r'Displacement ($u$)', rotation=-90, labelpad = 16, fontdict = {"size":19})

cbare_ax = fig.add_axes([0.835, 0.12, 0.025, 0.23])
# cbare_ax.set_xticks([])
# cbare_ax.yaxis.tick_right()
# cbare_ax.tick_params(axis='both',which='major',labelsize=17)
# cbare_ax.set_ylabel(r'Error',fontsize=18,rotation=-90)
# cbare_ax.yaxis.set_label_position("right")
# cbare_ax.set_ylim(-3,3)
cbare = plt.colorbar(cmape,cax=cbare_ax,shrink=0.95,extend='both',format='%+.1f')
cbare.ax.tick_params(labelsize=15)
cbare.ax.set_ylabel(r'Error', rotation=-90, labelpad = 16, fontdict = {"size":19})

# cbar_u = plt.colorbar(cmapu,ax=[axesr[-1],axesp[-1]],shrink=0.95,extend='both',format='%+.1f')
# cbar_e = plt.colorbar(cmape,ax=axese[-1],shrink=0.95,extend='both',format='%+.1f')


for t in range(5):
    axr = axesr[t];axp = axesp[t];axe = axese[t]
    for ax in (axr,axp,axe):
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
    T = 0.25*t
    axr.set_title(f'Prediction for time {T:.2f} (s)')
    x_test = torch.hstack((XY,T*torch.ones(XY.shape[0],1)))
    x_test_c = x_test.clone()
    x_test_c.requires_grad = True
    u_predict = model(x_test_c).detach().numpy().reshape(101,101)
    x,y = getxy(x_test)
    u_real = f_real(x_test[:,[0]],x_test[:,[1]],x_test[:,[2]]).detach().numpy().reshape(101,101)
    err = 2*(u_real-u_predict)/(np.abs(u_real)+1)
    print(max(np.abs(err.min()),np.abs(err.max())))

    axr.contourf(x,y,u_real,cmap=cm.turbo,levels=levelsu,alpha=0.9)
    axp.contourf(x,y,u_predict,cmap=cm.turbo,levels=levelsu,alpha=0.9)
    axe.contourf(x,y,err,cmap=cm.seismic,levels=levelse,alpha=0.9)
fig.savefig(f'Figures/{model.name}/Images/Evolution{model.n}/Results.png',bbox_inches='tight')
fig.savefig(f'Figures/{model.name}/Images/Evolution{model.n}/Results.pdf',bbox_inches='tight')

# for t in range(0,101):
#     T = 0.01*t
#     print(f'{t:03d}->{T:.3f}')
#     x_test = torch.hstack((XY,T*torch.ones(XY.shape[0],1)))
#     x_test_c = x_test.clone()
#     x_test_c.requires_grad = True
#     u_predict = model(x_test_c).detach().numpy()
#     x,y = getxy(x_test)
#     u_real = f_real(x_test[:,[0]],x_test[:,[1]],x_test[:,[2]]).detach().numpy()

#     fig, axes = plt.subplots(nrows=3,figsize=(8,9))
#     fig.suptitle(f'Prediction for time {T:.3f} (s)')
#     fig.subplots_adjust(hspace=0.2, wspace=0.2)
#     for i,ax in enumerate(axes):
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_xlim([xmin, xmax])
#         ax.set_ylim([ymin, ymax])
#         ax.set_title(Titles[i])
#         for key, spine in ax.spines.items():
#             if key in ['right','top','left','bottom']:
#                 spine.set_visible(False)
#         ax.axis('equal')
#     axu = axes[0];axv = axes[1];axp = axes[2]

#     cf = axu.scatter(x,y,c=u_real, alpha=0.5, edgecolors='none', cmap='rainbow', marker='o', s=2,vmin=-1.5,vmax=1.5)
#     divider = make_axes_locatable(axu)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     fig.colorbar(cf, cax=cax)#, fraction=0.046, pad=0.04)

#     cf = axv.scatter(x,y,c=u_predict, alpha=0.5, edgecolors='none', cmap='rainbow', marker='o', s=2,vmin=-1.5,vmax=1.5)
#     divider = make_axes_locatable(axv)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     fig.colorbar(cf, cax=cax)

#     cf = axp.scatter(x,y,c=u_real-u_predict, alpha=0.5, edgecolors='none', cmap='rainbow', marker='o', s=2,vmin=-3,vmax=3)
#     divider = make_axes_locatable(axp)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     fig.colorbar(cf, cax=cax)
#     fig.savefig(f'Figures/{model.name}/Images/Evolution{model.n}/Time{t:03d}.png',bbox_inches='tight')
#     plt.close(fig)
