import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from pyDOE import lhs
from torch import autograd
from tqdm import tqdm
import matplotlib.patheffects as patheffects

warnings.filterwarnings("ignore")
try:plt.style.use('../figstyle.mplstyle')
except:pass

from PINNPDE import FCN

text_style = dict(horizontalalignment='center', verticalalignment='center',
                    fontsize=12, fontfamily='monospace')

'Initial set up'
torch.manual_seed(1234)
np.random.seed(1234)
steps = 2000
lr = 0.075
layers = np.array([1,40,40,40,1])
Nf = 2500
tmin, tmax = 0., 2*np.pi
total_points_t = 1200

k = 1;m = 1;w0 = np.sqrt(k/m)
I0 = [[x,(2*np.random.randint(0,2)-1)*np.sqrt(1-x**2)] for x in 2*np.random.random(5)-1]
I0 += [[1,0],[-1,0],[0,1],[0,-1]]
I0 = [I0[i] for i in(5,0,6,1,2,3,7,4,8)]

t_test = torch.linspace(tmin,tmax,total_points_t).view(-1,1)
t_test_g = torch.linspace(tmin,tmax,total_points_t).view(-1,1)

# fig, axes = plt.subplots(3,3,sharex=True,sharey=True,figsize=(10,15))
# fig.subplots_adjust(hspace=0.09,wspace=0.025)
# fig2, axes2 = plt.subplots(3,3,sharex=True,sharey=True,figsize=(10,15))
# axes = axes.reshape(-1);axes2 = axes2.reshape(-1)
# axes[0].set_xlim(tmin,tmax);axes2[0].set_xlim(tmin,tmax)
# axes[0].set_ylim(-1.2,1.2);axes2[0].set_ylim(-0.1,0.6)
# axes[7].set_xlabel('Time (t) [a.u]',fontsize=20);axes[3].set_ylabel('Displacement (x) [a.u]',fontsize=20)
# axes2[7].set_xlabel('Time (t) [a.u]');axes2[3].set_ylabel('Energy (E) [a.u]')

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- NumPy ndarrays with the same shape.
    https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """
    average = np.average(values, weights=weights,axis=0)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights,axis=0)
    return (average, np.sqrt(variance))

'Loss Function'
def ResPDE(Tpde,Xpred):
    x_t = autograd.grad(Xpred,Tpde,torch.ones([Tpde.shape[0],1]).to(model.device), retain_graph=True, create_graph=True)[0]
    x_tt = autograd.grad(x_t,Tpde,torch.ones([Tpde.shape[0],1]).to(model.device), retain_graph=True, create_graph=True)[0]

    res = x_tt + w0*Xpred
    return [res]

'Initial Condition'
def InitialPos(Tcdt,Xcdt):
    x0 = X0*torch.ones(Xcdt.shape[0],1).to(model.device)
    return model.criterion(Xcdt,x0)
def InitialVel(Tcdt,Xcdt):
    v = autograd.grad(Xcdt,Tcdt,torch.ones([Tcdt.shape[0],1]).to(model.device),retain_graph=True, create_graph=True)[0]
    v0 = V0*torch.ones(v.shape[0],1).to(model.device)
    return model.criterion(v,v0)

'Exact Solution'
def Solution(t):
    return X0*np.cos(w0*t)+(V0/w0)*np.sin(w0*t)

#? Domain bounds
lb=t_test[0] #first value
ub=t_test[-1] #last value 

t_train= lb + (ub-lb)*lhs(1,Nf)
t_train = torch.vstack((t_train,t_test[0]))
t_test_np = np.linspace(tmin,tmax,total_points_t)
Results = {}
t_test_g.requires_grad = True

Times = []

for nnn in range(2):
    fig, axes = plt.subplots(2,2,sharex=True,sharey=True,figsize=(16,10))
    fig.subplots_adjust(hspace=0.09,wspace=0.025)
    axes = axes.reshape(-1)
    axes[0].set_xlim(tmin,tmax)
    axes[0].set_ylim(-1.2,1.2)
    axes[3].set_xlabel('Time (t) [a.u]',fontsize=20);axes[0].set_ylabel('Displacement (x) [a.u]',fontsize=20)

    axes[2].set_xlabel('Time (t) [a.u]',fontsize=20);axes[2].set_ylabel('Displacement (x) [a.u]',fontsize=20)
    for i in range(4):
        ax = axes[i]
        i += 5*nnn
        element = I0[i]
        X0, V0 = element
        print(f'x0={X0:.3f}, v0={V0:.3f}')
        x_test = Solution(t_test)
        BLoss = []
        Preds = []
        for j in tqdm(range(50)):
            model = FCN(layers,ResPDE,lr=lr,numequ=1,numcdt=2,name=f'SimpleHarmonicOscillator_DP{i+1:d}')
            model.n = j+1
            t_train = t_train.float().to(model.device)

            CDT = [(t_test[0].view(-1,1).float().to(model.device),InitialPos,3),\
                    (t_test[0].view(-1,1).float().to(model.device),InitialVel,3)]

            # fig3, ax3 = model.Train(t_train,CDT,nEpochs=steps,nLoss=100,Test=[t_test,x_test])
            # model.save()
            model.load_state_dict(torch.load(f'Models/{model.name}/Best/StateDict_{model.n}.model'))
            # plt.close(fig3)
            x_pred = model(t_test_g)
            BLoss.append(model.criterion(x_pred,x_test).detach().numpy())
            x_pred = x_pred.detach().numpy()
            ax.plot(t_test_np,x_pred,'b--',alpha=0.2,zorder=1)
            # Times.append(model.time)
            Preds.append(x_pred)
        BLoss = np.array(BLoss)
        ax.set_title(rf'$x_0=${X0:.3f}; $v_0=${V0:.3f}',fontsize=16)
        Preds = np.array(Preds)
        pred_m=[];pred_s=[]
        for l in range(Preds.shape[1]):
            a,s = weighted_avg_and_std(Preds[:,l,0],np.exp(-BLoss))
            pred_m.append(a);pred_s.append(s)
        pred_m = np.array(pred_m);pred_s = np.array(pred_s)
        print(Preds.shape,np.exp(-BLoss).shape)
        # pred_m,pred_s = weighted_avg_and_std(Preds,np.exp(-BLoss))
        ax.plot(t_test_np,x_test,'k-',zorder=3)
        ax.fill_between(t_test_np,pred_m-pred_s,pred_m+pred_s,color='m',alpha=0.4,zorder=2)
        ax.plot(t_test_np,pred_m,'m:',zorder=2)
        print()
        ax.text(np.pi,-1.1,fr'$\langle L_T\rangle=${np.mean(BLoss):.3E}',horizontalalignment='center', verticalalignment='center',
                color='red',path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],fontsize=15)
        ax.tick_params(axis='both',which='major',labelsize=14)
    fig.savefig(f'Figures/SHOComp{nnn}.pdf',bbox_inches='tight')
    fig.savefig(f'Figures/SHOComp{nnn}.png',bbox_inches='tight')
    plt.close()
