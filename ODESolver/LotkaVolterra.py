import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from pyDOE import lhs
from torch import autograd
from tqdm import tqdm
import matplotlib
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as patheffects
import os
# from Sophia import SophiaG

warnings.filterwarnings("ignore")
try:plt.style.use('../figstyle.mplstyle')
except:pass

from PINNPDE import FCN

def ColorMapMaker(VMin,VMax,map):
    norm = matplotlib.colors.Normalize(vmin=VMin, vmax=VMax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=map)
    return norm, mapper

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

cmap = get_cmap(11)

'Initial set up'
torch.manual_seed(1234)
np.random.seed(1234)

lr = 0.01
layers = np.array([1,25,25,25,20,25,25,25,2])


alpha, beta, delta, gamma = 2/3, 4/3, 1., 1.
Initial = np.linspace(0.9,1.8,10)
Vs = delta*Initial - gamma*np.log(Initial) + beta*Initial - alpha*np.log(Initial)
norm, mapper = ColorMapMaker(Vs.min(),Vs.max(),'winter')

def Transform(T,F):
    return 8*torch.sigmoid(F)

def ResPDE(Tpde,Fpde):
    xpde, ypde = Fpde.T
    xpde = xpde.view(-1,1)
    ypde = ypde.view(-1,1)
    x_t = autograd.grad(xpde,Tpde,torch.ones([xpde.shape[0],1]).to(model.device),retain_graph=True,create_graph=True)[0]
    y_t = autograd.grad(ypde,Tpde,torch.ones([xpde.shape[0],1]).to(model.device),retain_graph=True,create_graph=True)[0]
    f1 = x_t - alpha*xpde + beta*xpde*ypde
    f2 = y_t - delta*xpde*ypde + gamma*ypde
    f3 = delta*xpde - gamma*torch.log(xpde) + beta*ypde - alpha*torch.log(ypde) - V
    return [f1,f2,f3]

'Initial condition'
def InitialPos(Tcdt,Fcdt):
    xcdt, ycdt = Fcdt.T
    X0 = x0*torch.ones(Tcdt.shape[0],1).to(model.device)
    Y0 = y0*torch.ones(Tcdt.shape[0],1).to(model.device)
    res = model.criterion(xcdt,X0)+model.criterion(ycdt,Y0)
    return res

fig, ax = plt.subplots(2,figsize=(16,20),sharex=True,sharey=True)
fig.subplots_adjust(hspace=0.01,wspace=0.025)
ax[0].set_aspect('equal', adjustable='box')
ax[0].set_xlim(0.,4.0)
ax[0].set_ylim(0.,2.6)
ax[0].set_ylabel('Predators',fontsize=30)
ax[0].tick_params(axis='both',which='major',labelsize=19)

ax[1].set_aspect('equal', adjustable='box')
ax[1].set_xlim(0.,4.0)
ax[1].set_ylim(0.,2.6)
ax[1].set_xlabel('Preys',fontsize=30);ax[1].set_ylabel('Predators',fontsize=30)
ax[1].tick_params(axis='both',which='major',labelsize=19)

fig4, axes4 = plt.subplots(5,2,figsize=(16,20),sharex=True,sharey=True)
fig4.subplots_adjust(hspace=0.09,wspace=0.025)
print(axes4.shape)
axes4[2,0].set_ylabel('Specimens Concentration [a.u]',fontsize=15)
axes4[4,0].set_xlabel('Evolution Time (t) [a.u]',fontsize=20)
axes4[4,1].set_xlabel('Evolution Time (t) [a.u]',fontsize=20)
axes4[4,0].plot([],[],'k--',label='Preys')
axes4[4,0].plot([],[],'k:',label='Predators')
axes4[4,0].legend(title=r'$\sigma=Tanh$')
axes4[4,1].plot([],[],'k-',label='Preys')
axes4[4,1].plot([],[],'k-.',label='Predators')
axes4[4,1].legend(title=r'$\sigma=Sin$')
axes4 = axes4.reshape(-1)
# plt.show()
# exit()
for i,pos in enumerate(Initial):
    tmin, tmax = 0., 20+10*i**1.05#(np.exp(0.25*i)-1)#
    Nf = int(7500*(tmax/20))
    steps = 10000+int(10000*(tmax-20)/(20+10*9**1.05))
    total_points_t = 2500
    t_test = torch.linspace(tmin,tmax,total_points_t).view(-1,1)


    #? Domain bounds
    lb=t_test[0] #first value
    ub=t_test[-1] #last value 

    t_train= lb + (ub-lb)*lhs(1,Nf)
    t_train = torch.vstack((t_train,t_test[0]))
    t_test_np = np.linspace(tmin,tmax,total_points_t)
    x0 = pos; y0 = pos
    V = Vs[i]
    print(x0,y0,V)
    # model = FCN(layers,ResPDE,Transform,lr=lr,numequ=3,numcdt=1,name=f'PredatorPreyTanI{i+1:d}',scheduler='StepLR',argschedu={'gamma':0.5,'step_size':1500})
    # CDT = [(t_test[0].view(-1,1).float().to(model.device),InitialPos,3)]
    # t_train = t_train.float().to(model.device)
    # fig2, ax2, fig3, ax3 = model.Train(t_train,CDT,Verbose=True,nEpochs=steps,nLoss=500,SH=True)
    # model.save()
    # plt.close(fig2);plt.close(fig3)
    model = torch.load(f'Models/PredatorPreyTanI{i+1:d}/Final/Trained_1.model')
    model.load_state_dict(torch.load(f'Models/{model.name}/Best/StateDict_{model.n}.model'))
    ltan = model.best_valid_loss
    Fpred = model(t_test)
    xpred, ypred = Fpred.T
    xpred = xpred.detach().numpy();ypred = ypred.detach().numpy()
    ax[0].plot(xpred,ypred,ls='-',alpha=0.8,c=cmap(i))
    ax[0].text(3.33,0.15+0.25*i,f'L={model.best_valid_loss:.3E}',horizontalalignment='left', verticalalignment='center',\
            path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],color=cmap(i),fontsize=20)
    axes4[i].plot(t_test_np,xpred,color=cmap(i),ls='--')
    axes4[i].plot(t_test_np,ypred,color=cmap(i),ls=':')

    # model = FCN(layers,ResPDE,Transform,activation=torch.sin,lr=lr,numequ=3,numcdt=1,name=f'PredatorPreySinI{i+1:d}',scheduler='StepLR',argschedu={'gamma':0.5,'step_size':1500})
    # CDT = [(t_test[0].view(-1,1).float().to(model.device),InitialPos,3)]
    # t_train = t_train.float().to(model.device)
    # fig2, ax2, fig3, ax3 = model.Train(t_train,CDT,Verbose=True,nEpochs=steps,nLoss=500,SH=True)
    # model.save()
    # plt.close(fig2);plt.close(fig3)
    model = torch.load(f'Models/PredatorPreySinI{i+1:d}/Final/Trained_1.model')
    model.load_state_dict(torch.load(f'Models/{model.name}/Best/StateDict_{model.n}.model'))
    lsin = model.best_valid_loss
    Fpred = model(t_test)
    xpred, ypred = Fpred.T
    xpred = xpred.detach().numpy();ypred = ypred.detach().numpy()
    ax[1].plot(xpred,ypred,ls='-',alpha=0.8,c=cmap(i))
    ax[1].text(3.33,0.15+0.25*i,f'L={model.best_valid_loss:.3E}',horizontalalignment='left', verticalalignment='center',\
            path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],color=cmap(i),fontsize=20)
    axes4[i].plot(t_test_np,xpred,color=cmap(i),ls='-')
    axes4[i].plot(t_test_np,ypred,color=cmap(i),ls='-.')
    axes4[i].tick_params(axis='both',which='major',labelsize=14)
    print(f'{(ltan-lsin)*100/ltan:.1f}\n')

fig.subplots_adjust(right=0.83)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar_ax.set_ylim([0,10]);cbar_ax.set_xlim([-0.5,0.5])
yticks = np.arange(10)+0.5;yticksname = []
for i in range(10):
    rect = plt.Rectangle(( -0.5,i), 1, 1, facecolor=cmap(i))
    cbar_ax.add_artist(rect)
    V= Vs[i]
    yticksname.append(f'{V:.4f}')
cbar_ax.set_xticks([])
cbar_ax.yaxis.set_minor_locator(MultipleLocator(1))
cbar_ax.set_yticks(yticks,yticksname)
cbar_ax.yaxis.tick_right()
cbar_ax.tick_params(axis='both',which='major',labelsize=17)
cbar_ax.set_ylabel('V',fontsize=30,rotation=-90)
cbar_ax.yaxis.set_label_position("right")
cbar_ax.grid(which='minor',axis='y',ls='-',color='k',alpha=1,linewidth=3)
cbar_ax.grid(which='major',axis='y',ls='none',color='k',alpha=1,linewidth=3)

ax[0].text(2,2.4,r'$\sigma=Tanh$',horizontalalignment='center', verticalalignment='center',\
            color='k',fontsize=40)

ax[1].text(2,2.4,r'$\sigma=Sin$',horizontalalignment='center', verticalalignment='center',\
            color='k',fontsize=40)

ax[0].text(0.25,2.4,r'$(1)$',horizontalalignment='center', verticalalignment='center',\
            color='k',fontsize=25,path_effects=[patheffects.withStroke(linewidth=1, foreground='red')])

ax[1].text(0.25,2.4,r'$(2)$',horizontalalignment='center', verticalalignment='center',\
            color='k',fontsize=25,path_effects=[patheffects.withStroke(linewidth=1, foreground='red')])

fig.savefig('Figures/PredatorPreyComp.png',bbox_inches='tight')
fig.savefig('Figures/PredatorPreyComp.pdf',bbox_inches='tight')
plt.close(fig)

fig4.subplots_adjust(right=0.83)
cbar_ax = fig4.add_axes([0.85, 0.15, 0.05, 0.7])
cbar_ax.set_ylim([0,10]);cbar_ax.set_xlim([-0.5,0.5])
yticks = np.arange(10)+0.5;yticksname = []
for i in range(10):
    rect = plt.Rectangle(( -0.5,i), 1, 1, facecolor=cmap(i))
    cbar_ax.add_artist(rect)
    V= Vs[i]
    yticksname.append(f'{V:.4f}')
cbar_ax.set_xticks([])
cbar_ax.yaxis.set_minor_locator(MultipleLocator(1))
cbar_ax.set_yticks(yticks,yticksname)
cbar_ax.yaxis.tick_right()
cbar_ax.tick_params(axis='both',which='major',labelsize=14)
cbar_ax.set_ylabel('V',fontsize=20,rotation=-90)
cbar_ax.yaxis.set_label_position("right")
cbar_ax.grid(which='minor',axis='y',ls='-',color='k',alpha=1,linewidth=3)
cbar_ax.grid(which='major',axis='y',ls='none',color='k',alpha=1,linewidth=3)

fig4.savefig('Figures/PredatorPreyCompS.png',bbox_inches='tight')
fig4.savefig('Figures/PredatorPreyCompS.pdf',bbox_inches='tight')
plt.close()


# fig, ax = plt.subplots(figsize=(16,10))

# ax.set_aspect('equal', adjustable='box')
# ax.set_xlim(0.,4.0)
# ax.set_ylim(0.,2.6)
# ax.set_xlabel('Preys',fontsize=20);ax.set_ylabel('Predators',fontsize=20)

# fig4, axes4 = plt.subplots(10,sharex=True,sharey=True)

# for i,pos in enumerate(Initial):
#     tmin, tmax = 0., 20+10*i**1.05#(np.exp(0.25*i)-1)#
#     Nf = int(7500*(tmax/20))
#     steps = 10000+int(10000*(tmax-20)/(20+10*9**1.05))
#     total_points_t = 2500
#     t_test = torch.linspace(tmin,tmax,total_points_t).view(-1,1)


#     #? Domain bounds
#     lb=t_test[0] #first value
#     ub=t_test[-1] #last value 

#     t_train= lb + (ub-lb)*lhs(1,Nf)
#     t_train = torch.vstack((t_train,t_test[0]))
#     t_test_np = np.linspace(tmin,tmax,total_points_t)
#     x0 = pos; y0 = pos
#     V = Vs[i]
#     print(x0,y0,V)
#     model = FCN(layers,ResPDE,Transform,activation=torch.sin,lr=lr,numequ=3,numcdt=1,name=f'PredatorPreySinI{i+1:d}',scheduler='StepLR',argschedu={'gamma':0.5,'step_size':1500})
#     CDT = [(t_test[0].view(-1,1).float().to(model.device),InitialPos,3)]
#     t_train = t_train.float().to(model.device)
#     fig2, ax2, fig3, ax3 = model.Train(t_train,CDT,Verbose=True,nEpochs=steps,nLoss=500,SH=True)
#     model.save()
#     plt.close(fig2);plt.close(fig3)
#     model.load_state_dict(torch.load(f'Models/{model.name}/Best/StateDict_{model.n}.model'))
#     Fpred = model(t_test)
#     xpred, ypred = Fpred.T
#     xpred = xpred.detach().numpy();ypred = ypred.detach().numpy()
#     ax.plot(xpred,ypred,ls='-',alpha=0.8,c=cmap(i))
#     ax.text(2.6,0.15+0.25*i,f'L={model.best_valid_loss:.3E}; V={V:.4f}',horizontalalignment='left', verticalalignment='center',\
#             path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],color=cmap(i),fontsize=18)
#     axes4[i].plot(t_test_np,xpred,color=cmap(i),ls='--')
#     axes4[i].plot(t_test_np,ypred,color=cmap(i),ls=':')
#     axes4[i].set_ylabel('Specimens Concentration [a.u]',fontsize=15)
#     axes4[i].tick_params(axis='both',which='major',labelsize=14)
# axes4[-1].plot([],[],'k--',label='Preys')
# axes4[-1].plot([],[],'k:',label='Predators')
# axes4[-1].set_xlabel('Evolution Time (t) [a.u]',fontsize=20)
# axes4[-1].legend()
# # divider = make_axes_locatable(ax)
# # cax = divider.append_axes('right', size='5%', pad=0.05)
# # fig.colorbar(cmap, cax=cax, orientation='vertical')
# ax.tick_params(axis='both',which='major',labelsize=14)
# fig.savefig('Figures/PredatorPreySin.png',bbox_inches='tight')
# fig.savefig('Figures/PredatorPreySin.pdf',bbox_inches='tight')
# fig4.savefig('Figures/PredatorPreySinS.png',bbox_inches='tight')
# fig4.savefig('Figures/PredatorPreySinS.pdf',bbox_inches='tight')
# plt.close()