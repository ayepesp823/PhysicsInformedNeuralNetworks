import torch
from torch import autograd

from matplotlib import pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split

import numpy as np
from pyDOE import lhs         #Latin Hypercube Sampling
from scipy import io
from scipy.stats import median_abs_deviation as MAD
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from time import time
import warnings
from os.path import exists
from os import makedirs, listdir, remove
warnings.filterwarnings("ignore")
try:plt.style.use('../figstyle.mplstyle')
except:pass

from PINNODE import FCN

#Set default dtype to float32
torch.set_default_dtype(torch.float)

#PyTorch random number generator
torch.manual_seed(1234)

# Random number generators in other libraries
np.random.seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device == 'cuda': 
    print(torch.cuda.get_device_name()) 

#! Tunning parameters

steps=5000
lr=1e-3
layers = np.array([1,50,50,20,50,50,1]) #5 hidden layers
# To generate new data:
min=-1
max=1
total_points=500
#Nu: Number of training points (2 as we onlt have 2 boundaries), # Nf: Number of collocation points (Evaluate PDE)
Nu=2;Nf=250

#! Functions

def f_BC(x):
    return 1-torch.abs(x)
def f_real(x):
    return torch.sin(np.pi*x)
def PDE(x):
    return -1*(np.pi**2)*torch.sin(np.pi*x)

xmin,xmax = -1,1

x = torch.linspace(xmin,xmax,total_points).view(-1,1) #prepare to NN
y = f_real(x)

BC_1=x[0,:]
BC_2=x[-1,:]
# Total Tpaining points BC1+BC2
all_train=torch.vstack([BC_1,BC_2])
#Select Nu points
idx = np.random.choice(all_train.shape[0], Nu, replace=False) 
x_BC=all_train[idx]
#Select Nf points
# Latin Hypercube sampling for collocation points 
x_PDE = BC_1 + (BC_2-BC_1)*lhs(1,Nf)
x_PDE = torch.vstack((x_PDE,x_BC)) 

y_BC=f_BC(x_BC).to(device)

#Store tensors to GPU

x_PDE=x_PDE.float().to(device)
x_BC=x_BC.to(device)

'Loss Functions'
#Loss BC
def lossBC(model,x_BC,y_BC):
    loss_BC=model.criterion(model.forward(x_BC),y_BC)
    return loss_BC
#Loss PDE
def lossPDE(model,x_PDE,PDE):
    g=x_PDE.clone()
    g.requires_grad=True #Enable differentiation
    f=model.forward(g)
    f_x=autograd.grad(f,g,torch.ones([x_PDE.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0] #first derivative
    f_xx=autograd.grad(f_x,g,torch.ones([x_PDE.shape[0],1]).to(device), create_graph=True)[0] #second derivative
    return model.criterion(f_xx,PDE(g))

def Loss(model,x_BC,y_BC,x_PDE,lossPDE,lossBC,PDE):
    loss_bc=lossBC(model,x_BC,y_BC)
    loss_pde=lossPDE(model,x_PDE,PDE)
    return loss_bc+loss_pde

optimizers = ['Adam', 'SGD', 'Adadelta', 'Adagrad']
colors = ['b','r','g','y']
fig,axes = plt.subplots(2,figsize=(16,10))
if exists('OtherResults/OptimizerTests.pkl'):
    with open('OtherResults/OptimizerTests.pkl','rb') as file:
        CompResults=pkl.load(file)
else:CompResults={}
for i,optimizer in enumerate(optimizers):
    torch.manual_seed(123)
    print(f'Considering optimizer: {optimizer}\n')
    if optimizer not in CompResults.keys():
        Y = []
        History = []
        CompResults[optimizer] = {'minLoss':[],'duration':[]}
        for _ in range(10):
            model = FCN(layers,Loss,lr=0.001,optimizer=optimizer)
            t0 = time()
            model.Train([x_BC,y_BC,x_PDE,lossPDE,lossBC,PDE],10000)
            CompResults[optimizer]['duration'].append(time()-t0)
            CompResults[optimizer]['minLoss'].append(np.min(model.loss_history))
            yh = model(x)
            Y.append(yh.detach().numpy())
            History.append(model.loss_history)
            del model, yh, t0
        CompResults[optimizer]['mminLoss'] = np.mean(CompResults[optimizer]['minLoss'])
        CompResults[optimizer]['dminLoss'] = MAD(CompResults[optimizer]['minLoss'])
        CompResults[optimizer]['mduration'] = np.mean(CompResults[optimizer]['duration'])
        CompResults[optimizer]['dduration'] = MAD(CompResults[optimizer]['duration'])
        yh = np.median(Y,axis=0).reshape((-1))
        mHistory = np.median(History,axis=0).reshape((-1))
        dyh = MAD(Y,axis=0).reshape((-1));dHistory = MAD(History,axis=0).reshape((-1))
        CompResults[optimizer]['mPrediction'] = yh
        CompResults[optimizer]['dPrediction'] = dyh
        CompResults[optimizer]['mLossHistory'] = mHistory
        CompResults[optimizer]['dLossHistory'] = dHistory
        if exists(f'OtherResults/OptimizerTests2.pkl'):
            remove('OtherResults/OptimizerTests2.pkl')
        with open('OtherResults/OptimizerTests2.pkl','wb') as file:  
            pkl.dump(CompResults,file) 
        del Y, History, dyh, dHistory, mHistory

    axes[0].fill_between(x.detach().numpy().reshape((-1)),CompResults[optimizer]['mPrediction']-CompResults[optimizer]['dPrediction'],
                        CompResults[optimizer]['mPrediction']+CompResults[optimizer]['dPrediction'],color=colors[i],alpha=.3)
    axes[0].plot(x.detach().numpy().reshape((-1)),CompResults[optimizer]['mPrediction'],color=colors[i],ls='-',label=optimizer)
    axes[1].fill_between(np.arange(10000)+1,CompResults[optimizer]['mLossHistory']-CompResults[optimizer]['dLossHistory'],
                        CompResults[optimizer]['mLossHistory']+CompResults[optimizer]['dLossHistory'],color=colors[i],alpha=.3)
    axes[1].plot(CompResults[optimizer]['mLossHistory'],color=colors[i],ls='-',label=optimizer)

axes[0].plot(x,f_real(x).detach().numpy(),color='k',label='Real u(x)')
axes[0].legend(frameon=False,);axes[0].set_xlabel('x',color='black');axes[0].set_ylabel('f(x)',color='black')
axes[1].set_xlabel('Training Epochs');axes[1].set_ylabel('loss');axes[1].set_yscale('log')
fig.savefig('Figures/OptimizerTestsLoss2.pdf',bbox_inches='tight')
fig.savefig('Figures/OptimizerTestsLoss2.png',bbox_inches='tight')
plt.draw();plt.pause(10);plt.close(fig)
CompResults = pd.DataFrame(CompResults).T
fig2, axes = plt.subplots(1,2,figsize=(16,10))
bar1=axes[0].bar(CompResults.index,CompResults['mminLoss'],yerr=CompResults['dminLoss'],width=0.4,color=colors,ecolor='black',capsize=3)
bar2=axes[1].bar(CompResults.index,CompResults['mduration'],yerr=CompResults['dduration'],width=0.4,color=colors,ecolor='black',capsize=3)
axes[0].set_ylabel('min(loss)'),axes[0].set_yscale('log');axes[1].set_ylabel('simulation time (s)')
fig2.savefig('Figures/OptimizerTestsTimeMin2.pdf',bbox_inches='tight')
fig2.savefig('Figures/OptimizerTestsTimeMin2.png',bbox_inches='tight')
plt.draw();plt.pause(10);plt.close(fig2)