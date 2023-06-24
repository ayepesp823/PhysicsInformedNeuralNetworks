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
steps = 40000
lr = 0.01
layers = np.array([2] + 8*[50] + [5])

Nu, Nf = 5000, 20000#? #Nu: number of training points of boundary #Nf: number of collocation points (to evaluate PDE in)

def transform(x,y):
    y[:,[1]] = (np.pi/2)*torch.sigmoid(y[:,[1]])
    y[:,[1]] = 5*torch.sin(y[:,[1]])
    return y

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

# model = FCN(layers,ResPDE,transform,lr=lr,numequ=7,numcdt=3,name='KovasznayFlow',scheduler='StepLR',argschedu={'gamma':0.25,'step_size':10000})


'Gen Data'

xmin, xmax = 0,1.1
ymin, ymax = 0,0.41

total_points_x = 800
total_points_y = 800

x = torch.linspace(xmin,xmax,total_points_x).view(-1,1)
y = torch.linspace(ymin,ymax,total_points_y).view(-1,1)

#? Create the mesh 
X,Y=torch.meshgrid(x.squeeze(1),y.squeeze(1))

x_test=torch.hstack((X.transpose(1,0).flatten()[:,None],Y.transpose(1,0).flatten()[:,None]))

#? Domain bounds
lb=x_test[0] #first value
ub=x_test[-1] #last value 

#? Inlet

U_max = 1.0

inlet_X = torch.hstack((X[0,:][:,None],Y[0,:][:,None]))
inlet_u = (4*U_max*inlet_X[:,[1]]*(0.41-inlet_X[:,[1]])/(0.41**2))
#todo aditional condition: v=0

def InletCDT(Xcdt,Ycdt):
    u,v,psi,p,s11,s22,s12 = TransformOutput(Xcdt,Ycdt)
    inlet_v = torch.zeros(Xcdt.shape[0],1).to(model.device)
    return model.criterion(u,inlet_u)+model.criterion(v,inlet_v)

#? Outlet

outlet_X = torch.hstack((X[-1,:][:,None],Y[-1,:][:,None]))
#todo condition p=0

def OutletCDT(Xcdt,Ycdt):
    u,v,psi,p,s11,s22,s12 = TransformOutput(Xcdt,Ycdt)
    outlet_p = torch.zeros(Xcdt.shape[0],1).to(model.device)
    return model.criterion(p,outlet_p)

#? Walls
#* Lower Wall
lower_wall = torch.hstack((X[:,0][:,None],Y[:,0][:,None]))
#* Upper Wall
upper_wall = torch.hstack((X[:,-1][:,None],Y[:,-1][:,None]))
#* Cylinder Wall
r = 0.05
theta = torch.linspace(0,2*np.pi,500).view(-1,1)
cylinder_wall = torch.hstack((r*torch.cos(theta)+0.2,r*torch.sin(theta)+0.2))

wall_X = torch.vstack((lower_wall,upper_wall,cylinder_wall))
#todo conditions at wall_X: u=0,v=0

def WallsCDT(Xcdt,Ycdt):
    u,v,psi,p,s11,s22,s12 = TransformOutput(Xcdt,Ycdt)
    wall_u = torch.zeros(Xcdt.shape[0],1).float().to(model.device)
    wall_v = torch.zeros(Xcdt.shape[0],1).float().to(model.device)
    return model.criterion(u,wall_u)+model.criterion(v,wall_v)


#? Collocation point for equation residual

def DelCylPT(XY_c, xc=0.0, yc=0.0, r=0.1):
    '''
    delete points within cylinder
    '''
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
    return XY_c[dst>r,:]

model = torch.load('Models/KovasznayFlow/Final/Trained_4.model')
model.focusmin = None

X_train_Nf = lb+(ub-lb)*lhs(2,Nf)

#* refined grid definition
refined_min = torch.tensor([0.1,0.1])
refined_delta = torch.tensor([0.2,0.2])
X_train_refined_Nf = refined_min+refined_delta*lhs(2,Nf)

X_train_Nf = torch.vstack((X_train_Nf,X_train_refined_Nf)).numpy()
#* remove points inside cylinder
X_train_Nf = torch.from_numpy(DelCylPT(X_train_Nf,0.2,0.2,0.05))
x_test = torch.from_numpy(DelCylPT(x_test.numpy(),0.2,0.2,0.05)).float().to(model.device)

inlet_X = inlet_X.float().to(model.device)
inlet_u = inlet_u.float().to(model.device)
outlet_X = outlet_X.float().to(model.device)
wall_X = wall_X.float().to(model.device)
X_train_Nf = torch.vstack((X_train_Nf,wall_X,inlet_X,outlet_X)).float().to(model.device)

#? Boundary Condition list

CDT = [(wall_X,WallsCDT,100.),\
        (inlet_X,InletCDT,10.),\
        (outlet_X,OutletCDT,10.)]

def getxy(XY):
    x = XY[:,[0]].numpy()
    y = XY[:,[1]].numpy()
    return x,y

#? DOMAIN GRAPHIC
wall_x,wall_y = getxy(wall_X)
inlet_x,inlet_y = getxy(inlet_X)
outlet_x,outlet_y = getxy(outlet_X)
cylinder_x,cylinder_y = getxy(cylinder_wall)
domain_x,domain_y = getxy(X_train_Nf)

fig, ax = plt.subplots()
ax.scatter(domain_x,domain_y,alpha=0.1,marker='.',color='k')
ax.scatter(wall_x,wall_y,alpha=0.1,marker='.',color='b')
ax.scatter(inlet_x,inlet_y,alpha=0.1,marker='.',color='r')
ax.scatter(outlet_x,outlet_y,alpha=0.1,marker='.',color='y')
ax.scatter(cylinder_x,cylinder_y,alpha=0.1,marker='.',color='g')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
for key, spine in ax.spines.items():
    if key in ['right','top','left','bottom']:
        spine.set_visible(False)
ax.axis('equal')
fig.savefig('Figures/KovasznayFlow/Images/Points.png',bbox_inches='tight')
plt.show()


# _,_,_,_ = model.Train(X_train_Nf,CDT,Verbose=True,nEpochs=steps,nLoss=2000,SH=True)
# print("--- %s seconds ---" % (model.time))
# model.save()
# plt.close('all')


model.load_state_dict(torch.load(f'Models/{model.name}/Best/StateDict_{model.n}.model'))

def getresults(results):
    out = []
    for result in results:
        out.append(result.detach().numpy())
    return out


def preprocess(dir='Data/FluentSol.mat'):
    '''
    Load reference solution from Fenics or Fluent
    '''
    data = io.loadmat(dir)

    X = data['x']
    Y = data['y']
    P = data['p']
    vx = data['vx']
    vy = data['vy']

    x_star = X.flatten()[:, None]
    y_star = Y.flatten()[:, None]
    p_star = P.flatten()[:, None]
    vx_star = vx.flatten()[:, None]
    vy_star = vy.flatten()[:, None]

    return x_star, y_star, vx_star, vy_star, p_star

x_FLUENT, y_FLUENT, u_FLUENT, v_FLUENT, p_FLUENT = preprocess()
x_test=torch.hstack((torch.from_numpy(x_FLUENT),torch.from_numpy(y_FLUENT)))

# x_test = torch.from_numpy(DelCylPT(x_test.numpy(),0.2,0.2,0.05)).float().to(model.device)
x_test_c = x_test.clone()
x_test_c.requires_grad = True
u,v,_,p,_,_,_ = TransformOutput(x_test_c,model(x_test_c))
u,v,p = getresults([u,v,p])
x,y = getxy(x_test)

normu, cmapu = ColorMapMaker(min(u.min(),u_FLUENT.min()),max(u.max(),u_FLUENT.max()),'rainbow')
normv, cmapv = ColorMapMaker(min(v.min(),v_FLUENT.min()),max(v.max(),v_FLUENT.max()),'rainbow')
normp, cmapp = ColorMapMaker(min(p.min(),p_FLUENT.min()),max(p.max(),p_FLUENT.max()),'rainbow')

erru = 2*(u_FLUENT-u)/(np.abs(u_FLUENT)+1)
errv = 2*(v_FLUENT-v)/(np.abs(v_FLUENT)+1)
errp = 2*(p_FLUENT-p)/(np.abs(p_FLUENT)+1)

maxe = 0
mine = 50
for err in (erru,errv,errp):
    if err.max()>maxe:maxe=err.max()
    if err.min()<mine:mine=err.min()
# maxe *= 1.2

normerrs, cmaperrs = ColorMapMaker(mine,maxe,'rainbow')

Titles = [r'$u$ (m/s)',r'$v$ (m/s)',r'$P$ (Pa)']
fig, axes = plt.subplots(nrows=3,ncols=3,figsize=(16,12))
fig.subplots_adjust(hspace=0.1, wspace=0.075)
for i,ax in enumerate(axes.reshape(-1)):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    # for key, spine in ax.spines.items():
    #     if key in ['right','top','left','bottom']:
    #         spine.set_visible(False)
    # ax.axis('equal')

for i in range(3):
    ax1 = axes[i,0]
    ax2 = axes[i,1]
    ax3 = axes[i,2]
    if i==0:
        ax1.set_title('Exact', fontdict = {"size":18})
        ax2.set_title('Predicted', fontdict = {"size":18})
        ax3.set_title('Errors', fontdict = {"size":18})
    # else:
    #     ax1.set_title(Titles[i])
    #     ax2.set_title(Titles[i])
    #     ax3.set_title(Titles[i])

#! EXACT SOLUTION

axu = axes[0,0];axv = axes[1,0];axp = axes[2,0]

cf = axu.scatter(x_FLUENT,y_FLUENT,c=cmapu.to_rgba(u_FLUENT), edgecolors='none', marker='s', s=16)
# divider = make_axes_locatable(axu)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(cf, cax=cax)#, fraction=0.046, pad=0.04)

cf = axv.scatter(x_FLUENT,y_FLUENT,c=cmapv.to_rgba(v_FLUENT), edgecolors='none', marker='s', s=16)
# divider = make_axes_locatable(axv)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(cf, cax=cax)

cf = axp.scatter(x_FLUENT,y_FLUENT,c=cmapp.to_rgba(p_FLUENT), edgecolors='none', marker='s', s=16)
# divider = make_axes_locatable(axp)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(cf, cax=cax)

#! PREDICTED SOLUTION
axu = axes[0,1];axv = axes[1,1];axp = axes[2,1]

cf = axu.scatter(x,y,c=cmapu.to_rgba(u), edgecolors='none', marker='s', s=16)
# divider = make_axes_locatable(axu)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(cf, cax=cax)#, fraction=0.046, pad=0.04)

cf = axv.scatter(x,y,c=cmapv.to_rgba(v), edgecolors='none', marker='s', s=16)
# divider = make_axes_locatable(axv)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(cf, cax=cax)

cf = axp.scatter(x,y,c=cmapp.to_rgba(p), edgecolors='none', marker='s', s=16)
# divider = make_axes_locatable(axp)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(cf, cax=cax)

#! ERRORS
axu = axes[0,2];axv = axes[1,2];axp = axes[2,2]

cf = axu.scatter(x,y,c=cmaperrs.to_rgba(erru), edgecolors='none', marker='s', s=16)
# divider = make_axes_locatable(axu)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(cf, cax=cax)#, fraction=0.046, pad=0.04)

cf = axv.scatter(x,y,c=cmaperrs.to_rgba(errv), edgecolors='none', marker='s', s=16)
# divider = make_axes_locatable(axv)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(cf, cax=cax)

cf = axp.scatter(x,y,c=cmaperrs.to_rgba(errp), edgecolors='none', marker='s', s=16)

print(np.abs(erru).mean()*100)
print(np.abs(errv).mean()*100)
print(np.abs(errp).mean()*100)
# divider = make_axes_locatable(axp)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(cf, cax=cax)

cbar_u = plt.colorbar(cmapu,ax=[axes[0,0],axes[0,1]],shrink=0.95,extend='both',format='%+.1f')
cbar_u.ax.tick_params(labelsize=18)
cbar_u.ax.set_ylabel(r'$u$ (m/s)', rotation=-90, labelpad=22, fontdict = {"size":18})

cbar_v = plt.colorbar(cmapv,ax=[axes[1,0],axes[1,1]],shrink=0.95,extend='both',format='%+.1f')
cbar_v.ax.tick_params(labelsize=18)
cbar_v.ax.set_ylabel(r'$v$ (m/s)', rotation=-90, labelpad=22, fontdict = {"size":18})

cbar_p = plt.colorbar(cmapp,ax=[axes[2,0],axes[2,1]],shrink=0.95,extend='both',format='%+.1f')
cbar_p.ax.tick_params(labelsize=18)
cbar_p.ax.set_ylabel(r'$P$ (Pa)', rotation=-90, labelpad=22, fontdict = {"size":18})

cbar_err = plt.colorbar(cmaperrs,ax=[axes[0,2],axes[1,2],axes[2,2]],shrink=0.95,extend='both',format='%+.1f')
cbar_err.ax.tick_params(labelsize=18)
cbar_err.ax.set_ylabel(r'Discrepancy', rotation=-90, labelpad=22, fontdict = {"size":18})

fig.savefig(f'Figures/{model.name}/Images/Results_{model.n}_alter.png',bbox_inches='tight')
fig.savefig(f'Figures/{model.name}/Images/Results_{model.n}_alter.pdf',bbox_inches='tight')

plt.close('all')

fig, (axin,axup,axdw,axcy,axot) = plt.subplots(5,figsize=(16,21),sharey=True)
fig.subplots_adjust(hspace=0.2, wspace=0.075)

for ax in (axin,axup,axdw,axcy,axot):
    ax.tick_params(labelsize=12)
    for key, spine in ax.spines.items():
        if key in ['right','top']:
            spine.set_visible(False)
# cbar_err = plt.colorbar(cmaperrs,ax=[axin,axup,axdw,axcy,axot],shrink=0.95,extend='both',format='%+.1f')
# cbar_err.ax.tick_params(labelsize=12)
# cbar_err.ax.set_ylabel(r'Discrepancy', rotation=-90, labelpad=22, fontdict = {"size":15})

axin.text(0.5,0.825,'INLET',color='r',horizontalalignment='center', verticalalignment='center',fontsize=16,transform = axin.transAxes)
axup.text(0.5,0.825,'UPPER WALL',color='b',horizontalalignment='center', verticalalignment='center',fontsize=16,transform = axup.transAxes)
axdw.text(0.5,0.825,'LOWER WALL',color='b',horizontalalignment='center', verticalalignment='center',fontsize=16,transform = axdw.transAxes)
axcy.text(0.5,0.825,'CYLINDER WALL',color='g',horizontalalignment='center', verticalalignment='center',fontsize=16,transform = axcy.transAxes)
axot.text(0.5,0.825,'OUTLET',color='y',horizontalalignment='center', verticalalignment='center',fontsize=16,transform = axot.transAxes)

axin.set_xlabel('y [m]', fontdict = {"size":15})
axup.set_xlabel('x [m]', fontdict = {"size":15})
axdw.set_xlabel('x [m]', fontdict = {"size":15})
axcy.set_xlabel(r'$\theta$ [rad]', fontdict = {"size":15})
axot.set_xlabel('y [m]', fontdict = {"size":15})

xs = [inlet_X,upper_wall,lower_wall,cylinder_wall,outlet_X]
ns = [1,0,0,2,1]

for i,ax in enumerate((axin,axup,axdw,axcy)):
    x_test_c = xs[i].clone()
    x_test_c.requires_grad = True
    u,v,_,p,_,_,_ = TransformOutput(x_test_c,model(x_test_c))
    real = torch.zeros(x_test_c.shape[0],1).float().to(model.device)
    if i == 0:
        erru= (2*(inlet_u-u)/(1+torch.abs(inlet_u))).detach().numpy()
        errv= (2*(real-v)/(1+torch.abs(real))).detach().numpy()
    else:
        erru= (2*(real-u)/(1+torch.abs(real))).detach().numpy()
        errv= (2*(real-v)/(1+torch.abs(real))).detach().numpy()
    
    if ns[i]!=2: x = xs[i][:,ns[i]].detach().numpy()
    else: x = theta.detach().numpy()
    ax.plot(x,erru,ls='-',color='k')
    ax.plot(x,errv,ls=':',color='k')
    ax.set_xlim(x.min(),x.max())
    print(np.abs(erru).mean()*100)
    print(np.abs(errv).mean()*100)


x = xs[-1][:,ns[-1]].detach().numpy()
x_test_c = xs[-1].clone()
x_test_c.requires_grad = True
u,v,_,p,_,_,_ = TransformOutput(x_test_c,model(x_test_c))
real = torch.zeros(x_test_c.shape[0],1).float().to(model.device)
err = (2*(real-p)/(1+torch.abs(real))).detach().numpy()
print(np.abs(err).mean()*100)
axot.plot(x,err,ls='--',color='k')
axot.set_xlim(x.min(),x.max())

axdw.plot([],[],'k-',label='u')
axdw.plot([],[],'k:',label='v')
axdw.plot([],[],'k--',label='P')

axdw.set_ylabel('Discrepancy', fontdict = {"size":15})

axdw.legend(frameon=False,title='Predicted Quantities')

fig.savefig(f'Figures/{model.name}/Images/Results_{model.n}_alter_errs.png',bbox_inches='tight')
fig.savefig(f'Figures/{model.name}/Images/Results_{model.n}_alter_errs.pdf',bbox_inches='tight')
# plt.show()