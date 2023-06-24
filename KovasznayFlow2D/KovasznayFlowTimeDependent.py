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
steps = 1000#0
lr = 0.01
layers = np.array([3] + 8*[45] + [5])

Nu, Nf = 2000, 30000#? #Nu: number of training points of boundary #Nf: number of collocation points (to evaluate PDE in)

def TransformOutput(Xpde,Fcomp):
    psi = Fcomp[:,0:1]
    p   = Fcomp[:,1:2]
    s11 = Fcomp[:,2:3]
    s22 = Fcomp[:,3:4]
    s12 = Fcomp[:,4:5]

    psi_x_y_t = autograd.grad(psi,Xpde,torch.ones([Xpde.shape[0],1]).to(model.device),\
                            retain_graph=True, create_graph=True)[0]
    u = psi_x_y_t[:,[1]];v = -psi_x_y_t[:,[0]]
    
    return u,v,psi,p,s11,s22,s12

'Loss Function'
def ResPDE(Xpde,Fcomp):
    mu = 0.02;rho = 1.0
    u,v,psi,p,s11,s22,s12 = TransformOutput(Xpde,Fcomp)

    u_x_y_t = autograd.grad(u,Xpde,torch.ones([Xpde.shape[0],1]).to(model.device),\
                            retain_graph=True, create_graph=True)[0]
    u_x = u_x_y_t[:,[0]];u_y = u_x_y_t[:,[1]];u_t = u_x_y_t[:,[2]]

    v_x_y_t = autograd.grad(v,Xpde,torch.ones([Xpde.shape[0],1]).to(model.device),\
                            retain_graph=True, create_graph=True)[0]
    v_x = v_x_y_t[:,[0]];v_y = v_x_y_t[:,[1]];v_t = v_x_y_t[:,[2]]

    s11_x_y_t = autograd.grad(s11,Xpde,torch.ones([Xpde.shape[0],1]).to(model.device),\
                            retain_graph=True, create_graph=True)[0]
    s11_x = s11_x_y_t[:,[0]]

    s12_x_y_t = autograd.grad(s12,Xpde,torch.ones([Xpde.shape[0],1]).to(model.device),\
                            retain_graph=True, create_graph=True)[0]
    s12_x = s12_x_y_t[:,[0]];s12_y = s12_x_y_t[:,[1]]

    s22_x_y_t = autograd.grad(s22,Xpde,torch.ones([Xpde.shape[0],1]).to(model.device),\
                            retain_graph=True, create_graph=True)[0]
    s22_y = s22_x_y_t[:,[1]]

    f_u = rho*u_t + rho*(u*u_x + v*u_y) - s11_x - s12_y
    f_v = rho*v_t + rho*(u*v_x + v*v_y) - s12_x - s22_y

    f_mass = u_x+v_y

    f_s11 = -p + 2*mu*u_x - s11
    f_s22 = -p + 2*mu*v_y - s22
    f_s12 = mu*(u_y+v_x) - s12

    f_p = p + (s11+s22)/2
    return [f_u, f_v, f_s11, f_s22, f_s12, f_p,f_mass]

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


'Model'
argsoptim = {'max_iter':20,'max_eval':None,'tolerance_grad':1e-11,
            'tolerance_change':1e-11,'history_size':100,'line_search_fn':'strong_wolfe'}

model = FCN(layers,ResPDE,optimizer='LBFGS',argsoptim=argsoptim,lr=lr,numequ=7,numcdt=3,name='CavityLidTime',\
            scheduler='StepLR',argschedu={'gamma':1.5,'step_size':100})

model.focusmin = None;model.n=1
'Gen Data'

xmin, xmax = 0.0,1.0
ymin, ymax = 0.0,1.0
tmin, tmax = 0.0,1.0

numx = 201
numy = 201
numt = 201

x = torch.linspace(xmin,xmax,numx).view(-1,1)
y = torch.linspace(ymin,ymax,numy).view(-1,1)
t = torch.linspace(tmin,tmax,numt).view(-1,1)

x_test = Make3DMesh(xmin,xmax,ymin,ymax,tmin,tmax,numx,numy,numt)

def ReduceBoundary(X,Nu=Nu):
    idx = np.random.choice(X.shape[0],Nu,replace=False)
    return X[idx,:]

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

left_wall = (torch.tensor([xmin,ymin,tmin]) + (torch.tensor([xmin,ymax,tmax])-torch.tensor([xmin,ymin,tmin]))*lhs(3,500)).float().to(model.device)
rigt_wall = (torch.tensor([xmax,ymin,tmin]) + (torch.tensor([xmax,ymax,tmax])-torch.tensor([xmax,ymin,tmin]))*lhs(3,500)).float().to(model.device)
lowr_wall = (torch.tensor([xmin,ymin,tmin]) + (torch.tensor([xmax,ymin,tmax])-torch.tensor([xmin,ymin,tmin]))*lhs(3,500)).float().to(model.device)
uppr_wall = (torch.tensor([xmin,ymax,tmin]) + (torch.tensor([xmax,ymax,tmax])-torch.tensor([xmin,ymax,tmin]))*lhs(3,500)).float().to(model.device)

first_corner = Make3DMesh(xmin,xmin,ymax,ymax,tmin,tmax,1,1,51)
second_corner = Make3DMesh(xmin,xmin,ymin,ymin,tmin,tmax,1,1,51)
third_corner = Make3DMesh(xmax,xmax,ymin,ymin,tmin,tmax,1,1,51)
fourth_corner = Make3DMesh(xmax,xmax,ymax,ymax,tmin,tmax,1,1,51)

no_slip_walls = torch.vstack((left_wall,rigt_wall,lowr_wall,first_corner,second_corner,third_corner,fourth_corner)).float().to(model.device)

CDT = [(no_slip_walls,NoSlip,3),\
        (uppr_wall,LidUCDT,2),\
        (uppr_wall,LidVCDT,2)]

#? Domain bounds
lb=x_test[0] #first value
ub=x_test[-1] #last value 

X_train_Nf = lb + (ub-lb)*lhs(3,Nf)
X_train_Nf = torch.vstack((X_train_Nf,uppr_wall,no_slip_walls)).float().to(model.device)

# start_time = time()
# fig, ax, fig2, ax2 = model.Train(X_train_Nf,CDT,Verbose=True,nEpochs=steps,nLoss=50,SH=True)
# print("--- %s seconds ---" % (time() - start_time))
# model.save()
# plt.close('all')

model.load_state_dict(torch.load(f'Models/{model.name}/Best/StateDict_{model.n}.model'))
if not exists(f'Figures/{model.name}/Images/Evolution{model.n}'):
    makedirs(f'Figures/{model.name}/Images/Evolution{model.n}')

def getresults(results):
    out = []
    for result in results:
        out.append(result.detach().numpy().reshape(numx,numy))
    return out

def getxy(XY):
    x = XY[:,[0]].numpy().reshape(numx,numy)
    y = XY[:,[1]].numpy().reshape(numx,numy)
    return x,y

levelsv = np.linspace(0,1,21)
levelsp = np.linspace(-0.044,0.044,21)

normv, cmapv = ColorMapMaker(0,1,'turbo')
normp, cmapp = ColorMapMaker(-4.4,4.4,'Greys')

fig, axes = plt.subplots(2,3,figsize=(16,7))
fig.subplots_adjust(left=0.17,right=0.83)
cbar_p_ax = fig.add_axes([0.14, 0.125, 0.025, 0.75])
cbar_v_ax = fig.add_axes([0.835, 0.125, 0.025, 0.75])
cbar_p = plt.colorbar(cmapp,cax=cbar_p_ax,shrink=0.95,extend='both',format='%+.1f')
cbar_p.ax.yaxis.set_ticks_position('left')
cbar_p.ax.yaxis.set_label_position('left')
cbar_p.ax.tick_params(labelsize=15)
cbar_p.ax.set_ylabel(r'$P$ (Pa) $\times10^{-2}$', rotation=90, labelpad=9, fontdict = {"size":18})

cbar_v = plt.colorbar(cmapv,cax=cbar_v_ax,shrink=0.95,extend='both',format='%+.1f')
cbar_v.ax.tick_params(labelsize=15)
cbar_v.ax.set_ylabel(r'$|\vec{V}|$ (m/s)', rotation=-90, labelpad=20, fontdict = {"size":18})


fig.subplots_adjust(hspace=0.09,wspace=0.03)
axes = axes.reshape(-1)

for t in range(6):
    ax = axes[t]
    T = 0.2*t
    x_test = Make3DMesh(xmin,xmax,ymin,ymax,T,T,numx,numy,1)
    x_test_c = x_test.clone()
    x_test_c.requires_grad = True
    u_pred,v_pred,_,p_pred,_,_,_ = TransformOutput(x_test_c,model(x_test_c))
    u_pred,v_pred,p_pred = getresults([u_pred,v_pred,p_pred])
    V_pred = np.sqrt((u_pred**2)+(v_pred**2))
    x_pred, y_pred = getxy(x_test)
    # ax.axis('equal')
    ax.set_xticks([])
    if t%3 ==0:ax.set_yticks([0.9,1.0])
    else: ax.set_yticks([])
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([0.9, ymax])
    ax.set_title(f'Prediction for time {T:.1f} (s)')

    cf = ax.contourf(x_pred,y_pred,V_pred,cmap=cm.turbo,levels=levelsv,zorder=2,alpha=0.9)
    ax.contour(x_pred,y_pred,V_pred,cmap=cm.turbo,levels=levelsv,zorder=2,linestyles='-.')

    print(max(np.abs(p_pred.min()),p_pred.max()))

    c = ax.contourf(x_pred,y_pred,p_pred,cmap=cm.Greys,levels=levelsp,zorder=1)
    ax.contour(x_pred,y_pred,p_pred,cmap=cm.Greys,levels=levelsp,zorder=3,linestyles='--',linewidth=2)
    skip = 4
    cq = ax.quiver(x_pred[::skip, ::skip], y_pred[::skip, ::skip], u_pred[::skip, ::skip], v_pred[::skip, ::skip],color='w',edgecolor='k',zorder=3,units='xy',scale=15,linewidth=0.3)

fig.savefig(f'Figures/{model.name}/Images/Results_{model.n}.png',bbox_inches='tight')
fig.savefig(f'Figures/{model.name}/Images/Results_{model.n}.pdf',bbox_inches='tight')

fig,ax = plt.subplots(figsize=(8,7))
fig.subplots_adjust(left=0.17,right=0.83)
cbar_p_ax = fig.add_axes([0.14, 0.125, 0.025, 0.75])
cbar_v_ax = fig.add_axes([0.835, 0.125, 0.025, 0.75])
cbar_p = plt.colorbar(cmapp,cax=cbar_p_ax,shrink=0.95,extend='both',format='%+.1f')
cbar_p.ax.yaxis.set_ticks_position('left')
cbar_p.ax.yaxis.set_label_position('left')
cbar_p.ax.tick_params(labelsize=15)
cbar_p.ax.set_ylabel(r'$P$ (Pa) $\times10^{-2}$', rotation=90, labelpad=9, fontdict = {"size":18})

cbar_v = plt.colorbar(cmapv,cax=cbar_v_ax,shrink=0.95,extend='both',format='%+.1f')
cbar_v.ax.tick_params(labelsize=15)
cbar_v.ax.set_ylabel(r'$|\vec{V}|$ (m/s)', rotation=-90, labelpad=20, fontdict = {"size":18})

ax.axis('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.set_title(f'Prediction for time {0:.1f} (s)')

x_test = Make3DMesh(xmin,xmax,ymin,ymax,0,0,numx,numy,1)
x_test_c = x_test.clone()
x_test_c.requires_grad = True
u_pred,v_pred,_,p_pred,_,_,_ = TransformOutput(x_test_c,model(x_test_c))
u_pred,v_pred,p_pred = getresults([u_pred,v_pred,p_pred])
V_pred = np.sqrt((u_pred**2)+(v_pred**2))
x_pred, y_pred = getxy(x_test)

cf = ax.contourf(x_pred,y_pred,V_pred,cmap=cm.turbo,levels=levelsv,zorder=2,alpha=0.9)
ax.contour(x_pred,y_pred,V_pred,cmap=cm.turbo,levels=levelsv,zorder=2,linestyles='-.')


c = ax.contourf(x_pred,y_pred,p_pred,cmap=cm.Greys,levels=levelsp,zorder=1)
ax.contour(x_pred,y_pred,p_pred,cmap=cm.Greys,levels=levelsp,zorder=3,linestyles='--',linewidth=2)
skip = 4
cq = ax.quiver(x_pred[::skip, ::skip], y_pred[::skip, ::skip], u_pred[::skip, ::skip], v_pred[::skip, ::skip],color='w',edgecolor='k',zorder=3,units='xy',scale=15,linewidth=0.3)

fig.savefig(f'Figures/{model.name}/Images/Results_{model.n}_example.png',bbox_inches='tight')
fig.savefig(f'Figures/{model.name}/Images/Results_{model.n}_example.pdf',bbox_inches='tight')
# x_cyl = torch.linspace(xmin,xmax,81)
# y_cyl = torch.linspace(ymin,ymax,81)
# x_cyl, y_cyl = torch.meshgrid(x_cyl,y_cyl)
# x_cyl = x_cyl.flatten()[:, None]
# y_cyl = y_cyl.flatten()[:, None]

# XY = torch.hstack((x_cyl,y_cyl))

# def getresults(results):
#     out = []
#     for result in results:
#         out.append(result.detach().numpy().reshape(81,81))
#     return out
# levels = np.linspace(-4,4,81)
# dt = 0.001

# fig2, axes2 = plt.subplots(3,2,figsize=(16,9))
# fig2.subplots_adjust(hspace=0.075, wspace=0.045)
# axes2 = axes2.reshape(-1)
# norm, cmap = ColorMapMaker(-4,4,'rainbow')

# cbar = plt.colorbar(cmap,ax=axes2,shrink=0.95,extend='both',format='%+.1f')
# cbar.ax.tick_params(labelsize=18)
# cbar.ax.set_ylabel(r'$P$', rotation=-90, labelpad=22, fontdict = {"size":18})

# for t in range(401):
#     T = dt*t
#     x_test = torch.hstack((XY,T*torch.ones(XY.shape[0],1)))
#     x_test_c = x_test.clone()
#     x_test_c.requires_grad = True
#     u,v,_,p,_,_,_ = TransformOutput(x_test_c,model(x_test_c))
#     u,v,p = getresults([u,v,p])
#     x,y = getxy(x_test)
#     x = x.reshape(81,81);y = y.reshape(81,81)
#     fig, ax = plt.subplots(1,figsize=(8,9))
#     fig.suptitle(f'Prediction for time {T:.2f} (s)')
#     fig.set_size_inches(11, 7)
#     div = make_axes_locatable(ax)
#     cax = div.append_axes('right', '5%', '5%')

#     cf = ax.contourf(x,y,p,alpha= 0.5, cmap=cm.viridis,levels=levels);
#     cb = fig.colorbar(cf, cax=cax)
#     cr = ax.contour(x,y,p, cmap=cm.viridis,vmin=-4,vmax=4);
#     skip = 2
#     cq = ax.quiver(x[::skip, ::skip], y[::skip, ::skip], u[::skip, ::skip], v[::skip, ::skip])
#     ax.set_xlabel('$x$')
#     ax.set_ylabel('$y$');
#     fig.savefig(f'Figures/{model.name}/Images/Evolution{model.n}/Time{t:03d}.png',bbox_inches='tight')
#     plt.close(fig)
#     if t%80==0:
#         ax2 = axes2[t//80]
#         ax2.set_xticks([])
#         ax2.set_yticks([])
#         ax2.set_title(f'Prediction for time {T:.3f} (s)')
#         ax2.contourf(x,y,p,alpha= 0.5, cmap=cm.rainbow,levels=levels)
#         ax2.contour(x,y,p, cmap=cm.rainbow,vmin=-4,vmax=4)
#         ax2.quiver(x[::skip, ::skip], y[::skip, ::skip], u[::skip, ::skip], v[::skip, ::skip])

# fig2.savefig(f'Figures/{model.name}/Images/Evolution{model.n}/ResultPlot.png',bbox_inches='tight')

# imgs = glob.glob(f'Figures/{model.name}/Images/Evolution{model.n}/*.png')
# imgs.sort()

# frames = []

# for i in imgs:
#     new_frame = Image.open(i)
#     frames.append(new_frame)

# frames[0].save(f'Figures/{model.name}/Images/Evolution{model.n}/Evolution.gif', format='GIF',
#                append_images=frames[1:],
#                save_all=True,
#                duration=1200, loop=0)
