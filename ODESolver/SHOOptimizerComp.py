import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from pyDOE import lhs
from torch import autograd
from tqdm import tqdm
from scipy.stats import median_abs_deviation as MAD
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
steps = 5000
lr = 1e-3
layers = np.array([1,50,50,20,50,50,1])
Nf = 300;Nu=50
tmin, tmax = -1,1
total_points_t = 1200

Phi0 = [0,np.pi/2,np.pi]

t_test = torch.linspace(tmin,tmax,total_points_t).view(-1,1)
t_test_g = torch.linspace(tmin,tmax,total_points_t).view(-1,1)

# fig, axes = plt.subplots(3,2,sharex='col',sharey='col',figsize=(16,19),gridspec_kw={'width_ratios':[12,4]})
# lossaxes = axes[:,0]; predictaxes = axes[:,1]
# lossaxes[0].set_title('Computed Loss History',fontsize=19)
# lossaxes[1].set_ylabel('Total Loss',fontsize=18)
# lossaxes[2].set_xlabel('Epochs',fontsize=18)
# for ax in lossaxes:
#     ax.set_xlim(1,steps)
#     ax.set_yscale('log')
#     ax.tick_params(labelsize=15)
# predictaxes[0].set_title('Predicted Solution',fontsize=19)
# predictaxes[1].set_ylabel('Displacement (x) [a.u]',fontsize=18)
# predictaxes[1].plot([],[],'k-',label='Analytical')
# predictaxes[1].legend(frameon=False,fontsize=18)
# predictaxes[2].set_xlabel('Time (t) [a.u]',fontsize=18)
# for i,ax in enumerate(predictaxes):
#     ax.set_xlim(tmin,tmax)
#     ax.set_ylim(-1.2,1.2)
#     ax.tick_params(labelsize=15)
#     ax.yaxis.set_major_formatter('{x:+.1f}')
#     ax.yaxis.set_label_position("right")
#     ax.yaxis.tick_right()
# fig.subplots_adjust(hspace=0.04,wspace=0.08)

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

    res = x_tt + (np.pi**2)*Xpred
    return [res]

'Exact Solution'
def Solution(t):
    return torch.sin(np.pi*t+phi0)

'Initial Condition'
def FCDT(Tcdt,Xcdt):
    return model.criterion(Xcdt,Solution(Tcdt))


#? Domain bounds
lb=t_test[0] #first value
ub=t_test[-1] #last value 

Xcdt = torch.stack((lb,ub))
print(lb,ub)

t_train= lb + (ub-lb)*lhs(1,Nf)
t_train = torch.vstack((t_train,t_test[0]))
idx = np.random.choice(t_train.shape[0],Nu,replace=False)
t_data = t_train[idx]

t_test_np = np.linspace(tmin,tmax,total_points_t)
Results = {}
t_test_g.requires_grad = True

optimizers = ['Adam', 'Adadelta', 'Adagrad','LBFGS']
colors = ['b','r','g','m']

argsoptim = {'max_iter':20,'max_eval':None,'tolerance_grad':1e-11,
            'tolerance_change':1e-11,'history_size':100,'line_search_fn':'strong_wolfe'}

Times = []
legends = []
PHI0T = ['0',r'$\frac{\pi}{2}$',r'$\pi$']


for i,phi0 in enumerate(Phi0):
    legends = []
    fig, (axp,axl) = plt.subplots(2,figsize=(16,12))
    axl.set_title('Computed Loss History',fontsize=19)
    axl.set_ylabel('Total Loss',fontsize=18)
    axl.set_xlabel('Epochs',fontsize=18)
    axl.set_xlim(1,steps)
    axl.set_yscale('log')
    axl.tick_params(labelsize=15)
    axp.set_title('Predicted Solution',fontsize=19)
    axp.set_ylabel('Displacement (x) [a.u]',fontsize=18)
    axp.plot([],[],'k-',label='Analytical')
    axp.legend(frameon=False,fontsize=18)
    axp.set_xlabel('Time (t) [a.u]',fontsize=18)
    axp.set_xlim(tmin,tmax)
    axp.set_ylim(-1.2,1.2)
    axp.tick_params(labelsize=15)
    axp.yaxis.set_major_formatter('{x:+.1f}')
    # axp.yaxis.set_label_position("right")
    # axp.yaxis.tick_right()
    fig.subplots_adjust(hspace=0.2)
    # axl = lossaxes[i];axp = predictaxes[i]
    print(rf'phi0={phi0:.3f}')
    
    x_test = Solution(t_test)
    for k,optimizer in enumerate(optimizers):
        print(optimizer)
        TLoss = []
        BLoss = []
        Preds = []
        Times = []
        for j in tqdm(range(25)):
            if optimizer !='LBFGS':
                # model = FCN(layers,ResPDE,lr=lr,optimizer=optimizer,numequ=1,numcdt=2,name=f'SHOOp{optimizer}{i+1:d}')
                # model.n = j+1
                model = torch.load(f'Models/SHOOp{optimizer}{i+1:d}/Final/Trained_{j+1}.model')
                t_train = t_train.float().to(model.device)
                t_data = t_data.float().to(model.device)
                CDT = [(Xcdt,FCDT,3),(t_data,FCDT,5)]
                # fig3, ax3 = model.Train(t_train,CDT,nEpochs=steps,nLoss=100,Test=[t_test,x_test],RE=False)
            else:
                # model = FCN(layers,ResPDE,lr=lr,optimizer=optimizer,argsoptim=argsoptim,numequ=1,numcdt=2,name=f'SHOOp{optimizer}{i+1:d}')
                # model.n = j+1
                model = torch.load(f'Models/SHOOp{optimizer}{i+1:d}/Final/Trained_{j+1}.model')
                t_train = t_train.float().to(model.device)
                t_data = t_data.float().to(model.device)
                CDT = [(Xcdt,FCDT,3),(t_data,FCDT,5)]
                # fig3, ax3 = model.Train(t_train,CDT,nEpochs=int(steps/20),nLoss=5,Test=[t_test,x_test],RE=False)

            # model.save()
            model.load_state_dict(torch.load(f'Models/{model.name}/Best/StateDict_{model.n}.model'))
            # plt.close(fig3)
            x_pred = model(t_test_g)
            BLoss.append(model.criterion(x_pred,x_test).detach().numpy())
            x_pred = x_pred.detach().numpy()
            Preds.append(x_pred)
            TLoss.append(model.loss_history['total_loss'])
            Times.append(model.time)
        BLoss = np.array(BLoss)
        Times = np.array(Times)
        TLoss = np.array(TLoss)
        Preds = np.array(Preds)
        pred_m = np.median(Preds,axis=0).reshape((-1))
        tloss_m = np.median(TLoss,axis=0).reshape((-1))
        pred_s = MAD(Preds,axis=0).reshape((-1));tloss_s = MAD(TLoss,axis=0).reshape((-1))
        # tloss_m=[];tloss_s=[]
        # for l in range(TLoss.shape[1]):
        #     a,s = weighted_avg_and_std(TLoss[:,l],np.exp(-BLoss))
        #     tloss_m.append(a);tloss_s.append(s)
        # tloss_m = np.array(tloss_m);tloss_s = np.array(tloss_s)

        # pred_m=[];pred_s=[]
        # for l in range(Preds.shape[1]):
        #     a,s = weighted_avg_and_std(Preds[:,l],np.exp(-BLoss))
        #     pred_m.append(a);pred_s.append(s)
        # pred_m = np.array(pred_m).T;pred_s = np.array(pred_s).T
        # pred_m,pred_s = weighted_avg_and_std(Preds,np.exp(-BLoss))
        if optimizer !='LBFGS':r=1
        else:r=20
        fill = axl.fill_between(r*np.array(list(range(1,len(tloss_m)+1))),tloss_m-tloss_s,tloss_m+tloss_s,color=colors[k],alpha=0.4,zorder=2)
        line = axl.plot(r*np.array(list(range(1,len(tloss_m)+1))),tloss_m,color=colors[k],ls=':',zorder=2)

        axp.fill_between(t_test_np,pred_m-pred_s,pred_m+pred_s,color=colors[k],alpha=0.4,zorder=2)
        axp.plot(t_test_np,pred_m,color=colors[k],ls=':',zorder=2)
        axp.text(0.75-0.25*i,0.6-0.125*k,rf'$\langle t_t\rangle:${np.mean(Times):.1f} s, $\langle L_T\rangle=${np.mean(BLoss):.3E}',transform = axp.transAxes,horizontalalignment='center', verticalalignment='center',fontsize=16,color=colors[k])
        legends.append((fill,line[0]))
    axp.plot(t_test_np,x_test,'k-',zorder=3)
#     ax2.plot(t_test_np,0.5*k*(x_test**2),'k-.')

#     print()
#     ax2.fill_between(t_test_np,0.5*k*pred_m*(pred_m-pred_s),0.5*k*pred_m*(pred_m+pred_s),color='m',alpha=0.1)
#     ax2.plot(t_test_np,0.5*k*(pred_m**2),'m:')
#     ax2.text(np.pi,-0.05,fr'$\langle L_T\rangle=${np.mean(BLoss):.3E}',horizontalalignment='center', verticalalignment='center')
#     ax.text(np.pi,-1.1,fr'$\langle L_T\rangle=${np.mean(BLoss):.3E}',horizontalalignment='center', verticalalignment='center',
#             color='red',path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],fontsize=15)
#     ax.tick_params(axis='both',which='major',labelsize=14)
    axp.text(0.5,round((1+(-1)**i*0.8)/2,1),r'$\phi_0$='+PHI0T[i],transform = axp.transAxes,horizontalalignment='center', verticalalignment='center',fontsize=20)

    axl.legend(legends,optimizers,title='Optimizer',frameon=False,ncol=2,columnspacing=0.5,loc='upper right', fontsize=15, title_fontsize=16)
    # print(np.mean(Times),np.std(Times))
    fig.savefig(f'Figures/SHOOptPhi{i}.pdf',bbox_inches='tight')
    fig.savefig(f'Figures/SHOOptPhi{i}.png',bbox_inches='tight')
    print()
# fig2.savefig(f'Figures/SHOEComp.pdf',bbox_inches='tight')
# fig2.savefig(f'Figures/SHOEComp.png',bbox_inches='tight')
# plt.show()
