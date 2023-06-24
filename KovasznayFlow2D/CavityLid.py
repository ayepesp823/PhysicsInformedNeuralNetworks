import glob
import pickle as pkl
import warnings
from os import listdir, makedirs, remove
from os.path import exists
from time import time

import matplotlib
import matplotlib.colors as mcol
import numpy as np
import pandas as pd
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from pyDOE import lhs  # Latin Hypercube Sampling
from scipy import io
from scipy.stats import median_abs_deviation as MAD
from sklearn.model_selection import train_test_split
from torch import autograd, nn
from tqdm import tqdm

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
lr = 0.01
layers = np.array([2] + 8*[45] + [5])
Nu, Nf = 2000, 20000


def TransformOutput(Xpde,Fcomp):
    psi = Fcomp[:,0:1]
    p   = Fcomp[:,1:2]
    s11 = Fcomp[:,2:3]
    s22 = Fcomp[:,3:4]
    s12 = Fcomp[:,4:5]

    psi_x_y = autograd.grad(psi,Xpde,torch.ones([Xpde.shape[0],1]).to(model.device),\
                            retain_graph=True, create_graph=True)[0]
    u = psi_x_y[:,[1]];v = -psi_x_y[:,[0]]
    
    return u,v,psi,p,s11,s22,s12

'Loss Function'
def ResPDE(Xpde,Fcomp):
    mu = 0.02;rho = 1.0
    u,v,psi,p,s11,s22,s12 = TransformOutput(Xpde,Fcomp)

    u_x_y = autograd.grad(u,Xpde,torch.ones([Xpde.shape[0],1]).to(model.device),\
                            retain_graph=True, create_graph=True)[0]
    u_x = u_x_y[:,[0]];u_y = u_x_y[:,[1]]

    v_x_y = autograd.grad(v,Xpde,torch.ones([Xpde.shape[0],1]).to(model.device),\
                            retain_graph=True, create_graph=True)[0]
    v_x = v_x_y[:,[0]];v_y = v_x_y[:,[1]]

    s11_x_y = autograd.grad(s11,Xpde,torch.ones([Xpde.shape[0],1]).to(model.device),\
                            retain_graph=True, create_graph=True)[0]
    s11_x = s11_x_y[:,[0]]

    s12_x_y = autograd.grad(s12,Xpde,torch.ones([Xpde.shape[0],1]).to(model.device),\
                            retain_graph=True, create_graph=True)[0]
    s12_x = s12_x_y[:,[0]];s12_y = s12_x_y[:,[1]]

    s22_x_y = autograd.grad(s22,Xpde,torch.ones([Xpde.shape[0],1]).to(model.device),\
                            retain_graph=True, create_graph=True)[0]
    s22_y = s22_x_y[:,[1]]

    f_u = rho*(u*u_x + v*u_y) - s11_x - s12_y
    f_v = rho*(u*v_x + v*v_y) - s12_x - s22_y

    f_mass = u_x+v_y

    f_s11 = -p + 2*mu*u_x - s11
    f_s22 = -p + 2*mu*v_y - s22
    f_s12 = mu*(u_y+v_x) - s12

    f_p = p + (s11+s22)/2
    return [f_u, f_v, f_s11, f_s22, f_s12, f_p,f_mass]

'Model'
argsoptim = {'max_iter':20,'max_eval':None,'tolerance_grad':1e-11,
            'tolerance_change':1e-11,'history_size':100,'line_search_fn':'strong_wolfe'}

# model = FCN(layers,ResPDE,optimizer='LBFGS',argsoptim=argsoptim,lr=lr,numequ=7,numcdt=3,name='CavityFlow',\
#             scheduler='StepLR',argschedu={'gamma':1.5,'step_size':100})

# model.focusmin = None
# model.n = 1 

'Den Data'

def ReduceBoundary(X,Nu=Nu):
    idx = np.random.choice(X.shape[0],Nu,replace=False)
    return X[idx,:]


xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

numx, numy = 201, 201

x = torch.linspace(xmin,xmax,numx).view(-1,1)
y = torch.linspace(ymin,ymax,numy).view(-1,1)

#? Create the mesh 
X,Y=torch.meshgrid(x.squeeze(1),y.squeeze(1))
x_test=torch.hstack((X.transpose(1,0).flatten()[:,None],Y.transpose(1,0).flatten()[:,None]))

#? Walls

def NoSlip(Xcdt,Ycdt):
    "Relative velocity of the fluid to the wall equal to zero"
    u,v,_,_,_,_,_ = TransformOutput(Xcdt,Ycdt)
    V = torch.sqrt((u**2)+(v**2))
    no_slip = torch.zeros(Xcdt.shape[0],1).float().to(model.device)
    return model.criterion(V,no_slip)

def LidUCDT(Xcdt,Ycdt):
    "The only velocity is the x component"
    u,_,_,_,_,_,_ = TransformOutput(Xcdt,Ycdt)
    u_lid = torch.ones(Xcdt.shape[0],1).float().to(model.device)
    return model.criterion(u,u_lid)

def LidVCDT(Xcdt,Ycdt):
    "The only velocity is the x component"
    _,v,_,_,_,_,_ = TransformOutput(Xcdt,Ycdt)
    v_lid = torch.zeros(Xcdt.shape[0],1).float().to(model.device)
    return model.criterion(v,v_lid)

model = torch.load('Models/CavityFlow/Final/Trained_1.model')

left_wall = (torch.tensor([xmin,ymin]) + (torch.tensor([xmin,ymax])-torch.tensor([xmin,ymin]))*lhs(2,150)).float().to(model.device)
rigt_wall = (torch.tensor([xmax,ymin]) + (torch.tensor([xmax,ymax])-torch.tensor([xmax,ymin]))*lhs(2,150)).float().to(model.device)
lowr_wall = (torch.tensor([xmin,ymin]) + (torch.tensor([xmax,ymin])-torch.tensor([xmin,ymin]))*lhs(2,150)).float().to(model.device)
uppr_wall = (torch.tensor([xmin,ymax]) + (torch.tensor([xmax,ymax])-torch.tensor([xmin,ymax]))*lhs(2,150)).float().to(model.device)

no_slip_walls = torch.vstack((left_wall,rigt_wall,lowr_wall,torch.tensor([[xmin,ymax],[xmin,ymin],[xmax,ymin],[xmax,ymax]]))).float().to(model.device)

CDT = [(no_slip_walls,NoSlip,3),\
        (uppr_wall,LidUCDT,2),\
        (uppr_wall,LidVCDT,2)]

#? Domain bounds
lb=x_test[0] #first value
ub=x_test[-1] #last value 

X_train_Nf = lb + (ub-lb)*lhs(2,Nf)
X_train_Nf = torch.vstack((X_train_Nf,uppr_wall,no_slip_walls)).float().to(model.device)
print(X_train_Nf.shape)

# start_time = time()
# fig, ax, fig2, ax2 = model.Train(X_train_Nf,CDT,Verbose=True,nEpochs=steps,nLoss=50,SH=True)
# print("--- %s seconds ---" % (time() - start_time))
# model.save()
# plt.close('all')

model.load_state_dict(torch.load(f'Models/{model.name}/Best/StateDict_{model.n}.model'))

fig, ax = plt.subplots(figsize=(16,10))
ax.axis('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
div = make_axes_locatable(ax)
caxv = div.append_axes('right', '5%', '5%')
caxp = div.append_axes('left', '5%', '5%')

x_test_c = x_test.clone()
x_test_c.requires_grad = True

u_pred,v_pred,_,p_pred,_,_,_ = TransformOutput(x_test_c,model(x_test_c))

def getresults(results):
    out = []
    for result in results:
        out.append(result.detach().numpy().reshape(numx,numy))
    return out

def getxy(XY):
    x = XY[:,[0]].numpy().reshape(numx,numy)
    y = XY[:,[1]].numpy().reshape(numx,numy)
    return x,y

u_pred,v_pred,p_pred = getresults([u_pred,v_pred,p_pred])
V_pred = np.sqrt((u_pred**2)+(v_pred**2))
x_pred, y_pred = getxy(x_test)

levelsv = np.linspace(0,1,21)
levelsp = np.linspace(-1,1,21)

normv, cmapv = ColorMapMaker(0,1,'turbo')

cf = ax.contourf(x_pred,y_pred,V_pred,cmap=cm.turbo,levels=levelsv,zorder=2,alpha=0.9)
ax.contour(x_pred,y_pred,V_pred,cmap=cm.turbo,levels=levelsv,zorder=2,linestyles='-.')
cbar_v = fig.colorbar(cf,cax=caxv)
cbar_v.ax.tick_params(labelsize=18)
cbar_v.ax.set_ylabel(r'$|\vec{V}|$ (m/s)', rotation=-90, labelpad=22, fontdict = {"size":18})

c = ax.contourf(x_pred,y_pred,p_pred,cmap=cm.Greys,levels=levelsp,zorder=1)
ax.contour(x_pred,y_pred,p_pred,cmap=cm.Greys,levels=levelsp,zorder=3,linestyles='--',linewidth=0.5)
cbar_p = fig.colorbar(c,cax=caxp)
caxp.yaxis.set_ticks_position('left')
caxp.yaxis.set_label_position('left')
cbar_p.ax.tick_params(labelsize=18)
cbar_p.ax.set_ylabel(r'$P$ (Pa)', rotation=90, labelpad=12, fontdict = {"size":18})

skip = 4
cq = ax.quiver(x_pred[::skip, ::skip], y_pred[::skip, ::skip], u_pred[::skip, ::skip], v_pred[::skip, ::skip],color='w',edgecolor='k',zorder=3,units='xy',scale=15,linewidth=0.3)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')


fig.savefig(f'Figures/{model.name}/Images/Results_{model.n}.png',bbox_inches='tight')
fig.savefig(f'Figures/{model.name}/Images/Results_{model.n}.pdf',bbox_inches='tight')