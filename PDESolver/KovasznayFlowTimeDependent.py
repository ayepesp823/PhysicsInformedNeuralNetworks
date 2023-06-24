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
lr = 0.01
layers = np.array([3] + 7*[50] + [5])

Nu, Nf = 750, 5000#? #Nu: number of training points of boundary #Nf: number of collocation points (to evaluate PDE in)

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

model = FCN(layers,ResPDE,transform,lr=lr,numequ=7,numcdt=4,name='KovasznayFlowTimeDependent',\
            scheduler='StepLR',argschedu={'gamma':0.1,'step_size':12500})

model.focusmin = None
'Gen Data'

xmin, xmax = 0,1.10
ymin, ymax = 0,0.41
tmin, tmax = 0,0.50

numx = 165
numy =  61
numt =  75

def DelCylPT(XY_c, xc=0.0, yc=0.0, r=0.1):
    '''
    delete points within cylinder
    '''
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c[:,[0,1]]])
    return XY_c[dst>r,:]

def ReduceBoundary(X,Nu=Nu):
    idx = np.random.choice(X.shape[0],Nu,replace=False)
    return X[idx,:]

def getxy(XY):
    x = XY[:,[0]].numpy()
    y = XY[:,[1]].numpy()
    return x,y

#? Initial Conditions

#todo xyt for initial conditions [x,y,tmin]
ic_X = Make3DMesh(xmin,xmax,ymin,ymax,tmin,tmin,numx,numy,1).numpy()
ic_X = torch.from_numpy(DelCylPT(ic_X,0.2,0.2,0.05))
ic_X = ReduceBoundary(ic_X,Nu=2*Nu)

#todo conditios at t=tmin=0; u=0,v=0
def InitialConditionCDT(Xcdt,Ycdt):
    u,v,psi,p,s11,s22,s12 = TransformOutput(Xcdt,Ycdt)
    ic_u = torch.zeros(Xcdt.shape[0],1).to(model.device)
    ic_v = torch.zeros(Xcdt.shape[0],1).to(model.device)
    return model.criterion(u,ic_u) + model.criterion(v,ic_v)

#? Inlet

#todo inlet condition parameters
Umax = 1.0
T = tmax*2

#todo xyt for initial conditions [xmin,y,t]
inb_X = Make3DMesh(xmin,xmin,ymin,ymax,tmin,tmax,1,2*numy,numt)
inb_X = ReduceBoundary(inb_X,Nu=2*Nu)

#todo initial condition for u->u(xmin,y,t)
inb_u = (4*Umax*inb_X[:,[1]]*(ymax-inb_X[:,[1]])*(torch.sin(2*np.pi*inb_X[:,[2]]/T + 3*np.pi/2)+1.0)/(ymax**2)).float().to(model.device)

#todo conditios at x=xmin=0; u=u(xmin,y,t),v=0
def InletCDT(Xcdt,Ycdt):
    u,v,psi,p,s11,s22,s12 = TransformOutput(Xcdt,Ycdt)
    inb_v = torch.zeros(Xcdt.shape[0],1).to(model.device)
    return model.criterion(u,inb_u)+model.criterion(v,inb_v)

#? Outlet

#todo xyt for initial conditions [xmax,y,t]
out_X = Make3DMesh(xmax,xmax,ymin,ymax,tmin,tmax,1,2*numy,numt)
out_X = ReduceBoundary(out_X,Nu=2*Nu)

#todo conditios at x=xmax=1.1; p=0
def OutletCDT(Xcdt,Ycdt):
    u,v,psi,p,s11,s22,s12 = TransformOutput(Xcdt,Ycdt)
    outlet_p = torch.zeros(Xcdt.shape[0],1).to(model.device)
    return model.criterion(p,outlet_p)

#? Walls
#* Lower Wall
lower_wall = Make3DMesh(xmin,xmax,ymin,ymin,tmin,tmax,2*numx,1,numt)
lower_wall = ReduceBoundary(lower_wall)
#* Upper Wall
upper_wall = Make3DMesh(xmin,xmax,ymax,ymax,tmin,tmax,2*numx,1,numt)
upper_wall = ReduceBoundary(upper_wall)
#* Cylincer Wall
r=0.05
theta = torch.linspace(0,2*np.pi,629).view(-1,1)

#todo xyt for cylinder wall [r*cos(theta),r*sin(theta),t], theta in [0,2*pi]
x_cyl = r*torch.cos(theta) + 0.2
y_cyl = r*torch.sin(theta) + 0.2
x_cyl = x_cyl.view(-1,1);y_cyl = y_cyl.view(-1,1)
cyl = torch.hstack((x_cyl,y_cyl))

t_cyl = torch.linspace(tmin,tmax,numt).view(-1,1)

cylinder_wall = torch.hstack((x_cyl,y_cyl,torch.zeros(x_cyl.shape[0],1)))

for t in range(1,51):
    t = 0.01*t
    cylinder_wall = torch.vstack((cylinder_wall,torch.hstack((x_cyl,y_cyl,t*torch.ones(x_cyl.shape[0],1)))))

cylinder_wall = ReduceBoundary(cylinder_wall)

x_cyl, y_cyl = getxy(cylinder_wall)


wall_X = torch.vstack((lower_wall,upper_wall,cylinder_wall)).float().to(model.device)

#todo conditions at cylinder wall; u=0,v=0
def WallsCDT(Xcdt,Ycdt):
    u,v,psi,p,s11,s22,s12 = TransformOutput(Xcdt,Ycdt)
    wall_u = torch.zeros(Xcdt.shape[0],1).float().to(model.device)
    wall_v = torch.zeros(Xcdt.shape[0],1).float().to(model.device)
    return model.criterion(u,wall_u)+model.criterion(v,wall_v)

#? Collocation points for equation residual

lb = torch.tensor([xmin,ymin,tmin])
ub = torch.tensor([xmax,ymax,tmax])

X_train_Nf = lb + (ub-lb)*lhs(3,Nf)

#* refined grid definition
refined_min1 = torch.tensor([xmin,0.1,tmin])
refined_delta1 = torch.tensor([xmax,0.2,tmax])
X_train_refined_Nf1 = refined_min1+refined_delta1*lhs(3,Nf)

refined_min2 = torch.tensor([0.1,ymin,tmin])
refined_delta2 = torch.tensor([0.2,ymax,tmax])
X_train_refined_Nf2 = refined_min2+refined_delta2*lhs(3,Nf)

X_train_Nf = torch.vstack((X_train_Nf,X_train_refined_Nf1,X_train_refined_Nf2)).numpy()

#* remove points inside cylinder
X_train_Nf = torch.from_numpy(DelCylPT(X_train_Nf,0.2,0.2,0.05))



#? Boundary Condition list

CDT = [(wall_X,WallsCDT,100.),\
        (inb_X,InletCDT,10.),\
        (out_X,OutletCDT,10.),\
        (ic_X,InitialConditionCDT,30.)]

#? DOMAIN GRAPHIC
wall_x,wall_y = getxy(wall_X)
inlet_x,inlet_y = getxy(inb_X)
outlet_x,outlet_y = getxy(out_X)
cylinder_x,cylinder_y = getxy(cylinder_wall)
domain_x,domain_y = getxy(X_train_Nf)

# plt.scatter(domain_x,domain_y,alpha=0.1,marker='.',color='k')
# plt.scatter(wall_x,wall_y,alpha=0.1,marker='.',color='b')
# plt.scatter(inlet_x,inlet_y,alpha=0.1,marker='.',color='r')
# plt.scatter(outlet_x,outlet_y,alpha=0.1,marker='.',color='y')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

X_train_Nf = torch.vstack((X_train_Nf,wall_X,inb_X,out_X,ic_X)).float().to(model.device)

start_time = time()
# fig, ax, fig2, ax2 = model.Train(X_train_Nf,CDT,Verbose=True,nEpochs=steps,nLoss=2000,SH=True)
# print("--- %s seconds ---" % (time() - start_time))
# model.save()
# plt.close('all')

model.n = 5

model.load_state_dict(torch.load(f'Models/{model.name}/Best/StateDict_{model.n}.model'))

x_cyl = torch.linspace(xmin,xmax,500)
y_cyl = torch.linspace(ymin,ymax,500)
x_cyl, y_cyl = torch.meshgrid(x_cyl,y_cyl)
x_cyl = x_cyl.flatten()[:, None]
y_cyl = y_cyl.flatten()[:, None]

XY = torch.hstack((x_cyl,y_cyl))
XY = torch.from_numpy(DelCylPT(XY.numpy(),0.2,0.2,0.05)).float().to(model.device)

def getresults(results):
    out = []
    for result in results:
        out.append(result.detach().numpy())
    return out

if not exists(f'Figures/{model.name}/Images/Evolution{model.n}'):
    makedirs(f'Figures/{model.name}/Images/Evolution{model.n}')

frames = []
Titles = [r'$u$ (m/s)',r'$v$ (m/s)',r'$P$ (Pa)']

fig2, axes2 = plt.subplots(nrows=3,ncols=3,figsize=(16,10))
fig2.subplots_adjust(hspace=0.1, wspace=0.075)

fig3, axes3 = plt.subplots(nrows=3,ncols=3,figsize=(16,10))
fig3.subplots_adjust(hspace=0.1, wspace=0.075)

for ax in axes2.reshape(-1):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
for ax in axes3.reshape(-1):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

for i in range(3):
    axes2[0,i].set_title(f'Prediction for time {0.1*i:.1f} (s)', fontdict = {"size":15})
    axes3[0,i].set_title(f'Prediction for time {0.1*i+0.3:.1f} (s)', fontdict = {"size":15})

normu, cmapu = ColorMapMaker(0,2.5,'rainbow')
normv, cmapv = ColorMapMaker(-1,1,'rainbow')
normp, cmapp = ColorMapMaker(0,2.2,'rainbow')

cbar_u = plt.colorbar(cmapu,ax=[axes2[0,0],axes2[0,1],axes2[0,2]],shrink=0.95,extend='both',format='%+.1f')
cbar_u.ax.tick_params(labelsize=18)
cbar_u.ax.set_ylabel(r'$u$ (m/s)', rotation=-90, labelpad=22, fontdict = {"size":18})
cbar_u = plt.colorbar(cmapu,ax=[axes3[0,0],axes3[0,1],axes3[0,2]],shrink=0.95,extend='both',format='%+.1f')
cbar_u.ax.tick_params(labelsize=18)
cbar_u.ax.set_ylabel(r'$u$ (m/s)', rotation=-90, labelpad=22, fontdict = {"size":18})

cbar_v = plt.colorbar(cmapv,ax=[axes2[1,0],axes2[1,1],axes2[1,2]],shrink=0.95,extend='both',format='%+.1f')
cbar_v.ax.tick_params(labelsize=18)
cbar_v.ax.set_ylabel(r'$v$ (m/s)', rotation=-90, labelpad=22, fontdict = {"size":18})
cbar_v = plt.colorbar(cmapv,ax=[axes3[1,0],axes3[1,1],axes3[1,2]],shrink=0.95,extend='both',format='%+.1f')
cbar_v.ax.tick_params(labelsize=18)
cbar_v.ax.set_ylabel(r'$v$ (m/s)', rotation=-90, labelpad=22, fontdict = {"size":18})

cbar_p = plt.colorbar(cmapp,ax=[axes2[2,0],axes2[2,1],axes2[2,2]],shrink=0.95,extend='both',format='%+.1f')
cbar_p.ax.tick_params(labelsize=18)
cbar_p.ax.set_ylabel(r'$P$ (Pa)', rotation=-90, labelpad=22, fontdict = {"size":18})
cbar_p = plt.colorbar(cmapp,ax=[axes3[2,0],axes3[2,1],axes3[2,2]],shrink=0.95,extend='both',format='%+.1f')
cbar_p.ax.tick_params(labelsize=18)
cbar_p.ax.set_ylabel(r'$P$ (Pa)', rotation=-90, labelpad=22, fontdict = {"size":18})

PREDICTS = {'INLET':{'u':[],'v':[],'p':[]},\
            'OUTLE':{'u':[],'v':[],'p':[]},\
            'UPPER':{'u':[],'v':[],'p':[]},\
            'LOWER':{'u':[],'v':[],'p':[]},\
            'CYLIN':{'u':[],'v':[],'p':[]}}


for t in range(0,101):
    T = 0.005*t
    print(f'{t:03d}->{T:.3f}')
    x_test = torch.hstack((XY,T*torch.ones(XY.shape[0],1)))
    x_test_c = x_test.clone()
    x_test_c.requires_grad = True
    u,v,_,p,_,_,_ = TransformOutput(x_test_c,model(x_test_c))

    u,v,p = getresults([u,v,p])
    x,y = getxy(x_test)

    fig, axes = plt.subplots(nrows=3,figsize=(8,9))
    fig.suptitle(f'Prediction for time {T:.3f} (s)')
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    for i,ax in enumerate(axes):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_title(Titles[i])
        for key, spine in ax.spines.items():
            if key in ['right','top','left','bottom']:
                spine.set_visible(False)
        ax.axis('equal')
    axu = axes[0];axv = axes[1];axp = axes[2]

    cf = axu.scatter(x,y,c=u, alpha=0.5, edgecolors='none', cmap='rainbow', marker='o', s=2,vmin=0,vmax=2.5)
    divider = make_axes_locatable(axu)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(cf, cax=cax)#, fraction=0.046, pad=0.04)

    cf = axv.scatter(x,y,c=v, alpha=0.5, edgecolors='none', cmap='rainbow', marker='o', s=2,vmin=-1,vmax=1)
    divider = make_axes_locatable(axv)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(cf, cax=cax)

    cf = axp.scatter(x,y,c=p, alpha=0.5, edgecolors='none', cmap='rainbow', marker='o', s=2,vmin=0,vmax=2.2)
    divider = make_axes_locatable(axp)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(cf, cax=cax)
    fig.savefig(f'Figures/{model.name}/Images/Evolution{model.n}/Time{t:03d}.png',bbox_inches='tight')
    plt.close(fig)

    if t in [0,20,40,60,80,100]:
        if t in [0,20,40]:
            axu = axes2[0,int(t/20)];axv = axes2[1,int(t/20)];axp = axes2[2,int(t/20)]
        if t in [60,80,100]:
            axu = axes3[0,int(t/20)-3];axv = axes3[1,int(t/20)-3];axp = axes3[2,int(t/20)-3]
        axu.scatter(x,y,c=cmapu.to_rgba(u), alpha=0.5, edgecolors='none', marker='s', s=16)
        axv.scatter(x,y,c=cmapv.to_rgba(v), alpha=0.5, edgecolors='none', marker='s', s=16)
        axp.scatter(x,y,c=cmapp.to_rgba(p), alpha=0.5, edgecolors='none', marker='s', s=16)
    #inletcop
    x_in = Make3DMesh(xmin,xmin,ymin,ymax,T,T,1,numy,1)
    x_test_c = x_in.clone()
    x_test_c.requires_grad = True
    u,v,_,p,_,_,_ = TransformOutput(x_test_c,model(x_test_c))
    u,v,p = getresults([u,v,p])
    PREDICTS['INLET']['u'].append(u)
    PREDICTS['INLET']['v'].append(v)
    PREDICTS['INLET']['p'].append(p)

    #outletcop
    x_ou = Make3DMesh(xmax,xmax,ymin,ymax,T,T,1,numy,1)
    x_test_c = x_ou.clone()
    x_test_c.requires_grad = True
    u,v,_,p,_,_,_ = TransformOutput(x_test_c,model(x_test_c))
    u,v,p = getresults([u,v,p])
    PREDICTS['OUTLE']['u'].append(u)
    PREDICTS['OUTLE']['v'].append(v)
    PREDICTS['OUTLE']['p'].append(p)
    
    #upper
    x_up = Make3DMesh(xmin,xmax,ymax,ymax,T,T,numx,1,1)
    x_test_c = x_up.clone()
    x_test_c.requires_grad = True
    u,v,_,p,_,_,_ = TransformOutput(x_test_c,model(x_test_c))
    u,v,p = getresults([u,v,p])
    PREDICTS['UPPER']['u'].append(u)
    PREDICTS['UPPER']['v'].append(v)
    PREDICTS['UPPER']['p'].append(p)
    
    #lower
    x_lw = Make3DMesh(xmin,xmax,ymin,ymin,T,T,numx,1,1)
    x_test_c = x_lw.clone()
    x_test_c.requires_grad = True
    u,v,_,p,_,_,_ = TransformOutput(x_test_c,model(x_test_c))
    u,v,p = getresults([u,v,p])
    PREDICTS['LOWER']['u'].append(u)
    PREDICTS['LOWER']['v'].append(v)
    PREDICTS['LOWER']['p'].append(p)
    
    #cyl
    x_cy = torch.hstack((cyl,T*torch.ones(cyl.shape[0],1)))
    x_test_c = x_cy.clone()
    x_test_c.requires_grad = True
    u,v,_,p,_,_,_ = TransformOutput(x_test_c,model(x_test_c))
    u,v,p = getresults([u,v,p])
    PREDICTS['CYLIN']['u'].append(u)
    PREDICTS['CYLIN']['v'].append(v)
    PREDICTS['CYLIN']['p'].append(p)


fig2.savefig(f'Figures/{model.name}/Images/Evolution{model.n}/ResultPanel1.png',bbox_inches='tight')
fig3.savefig(f'Figures/{model.name}/Images/Evolution{model.n}/ResultPanel2.png',bbox_inches='tight')
plt.close('all')

for bound in PREDICTS.keys():
    for pred in PREDICTS[bound].keys():
        PREDICTS[bound][pred] = np.array(PREDICTS[bound][pred])
        print(bound,pred,PREDICTS[bound][pred].shape)
    print()

with open(f'Models/{model.name}/Results{model.n}.pkl','wb') as fp:
    pkl.dump(PREDICTS,fp)

# fig2.savefig(f'Figures/{model.name}/Images/Evolution{model.n}/ResultPanel1.pdf',bbox_inches='tight')
# fig3.savefig(f'Figures/{model.name}/Images/Evolution{model.n}/ResultPanel2.pdf',bbox_inches='tight')


