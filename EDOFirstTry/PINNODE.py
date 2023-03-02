"""Script inspired by https://github.com/jdtoscano94/Learning-Python-Physics-Informed-Machine-Learning-PINNs-DeepONets repo"""

import torch
from torch import  autograd
from torch import Tensor
from torch import nn
from torch import optim

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
import warnings
from os.path import exists
from os import makedirs, listdir, remove
warnings.filterwarnings("ignore")
try:plt.style.use('../figstyle.mplstyle')
except:pass

#? lrfinder
from torch_lr_finder import LRFinder

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

#! Neural Network

class FCN(nn.Module):
    #?Neural Network
    def __init__(self,layers,loss_function,activation=nn.Tanh(),lr=0.5,criterion=nn.MSELoss(reduction ='mean'),optimizer='Adam',scheduler=None,argsoptim={},argschedu={}):
        super().__init__() #call __init__ from parent class 
        'activation function'
        self.activation = activation
        'criterion'#? used for estimation of each individual loss (PDE and BC)
        self.criterion = criterion#? this one computes loss using the estimated curve
        'loss function'#? used for estimation of general loss (PDE+BC)
        self.loss_function = loss_function
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]) 
        self.iter = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        'Xavier Normal Initialization'
        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(layers)-1):
            
            # weights from a normal distribution with 
            # Recommended gain value for tanh = 5/3?
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)   
        self.to(self.device)
        try:
            self.optimizer = getattr(torch.optim, optimizer)(self.parameters(), lr=lr, **argsoptim)
        except:
            self.optimizer = torch.optim.Adam(self.parameters(),lr=lr,**argsoptim)
            print(f'\nOptimizer "{optimizer}" was not found, Adam was used instead\n')
        try:
            self.scheduler = getattr(torch.optim.lr_scheduler,scheduler)(self.optimizer,**argschedu)
        except:
            self.scheduler = None
            if scheduler is not None:print(f'Learning Rate Scheduler "{scheduler}" was not found. Ingoring option.')
    'foward pass'
    def forward(self,x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                
        a = x.float()
        for i in range(len(layers)-2):  
            z = self.linears[i](a)              
            a = self.activation(z)    
        a = self.linears[-1](a)
        return a
    def loss(self,args):
        return self.loss_function(self,*args)

    def Train(self,lossargs,nEpochs=100,BatchSize=None,history=True,Verbose=False,nLoss=1000,pScheduler=True):
        if history:self.loss_history = []
        if self.scheduler is not None: 
            self.schhist=None
        iterator = range(nEpochs)
        if Verbose: iterator = tqdm(iterator)
        for n in iterator:
            self.optimizer.zero_grad()
            Loss = self.loss(lossargs)
            Loss.backward()
            self.optimizer.step()
            if self.scheduler is not None: 
                if pScheduler:
                    if self.schhist is not None:self.schhist.append(self.optimizer.param_groups[0]["lr"])
                    else:self.schhist = [self.optimizer.param_groups[0]["lr"]]
                try:self.scheduler.step()
                except:self.scheduler.step(Loss)
            if Verbose and n%nLoss==0:
                print(f'\nloss at iteration {n}: {Loss:.3e}')
            if history:self.loss_history.append(Loss.detach().cpu().numpy())

    def save(self,name):
        if not exists('Models'):
            makedirs('Models')
        print(name)
        otherm = [int(file.replace(name,'').replace('.model','')) for file in listdir('Models') if file.startswith(name)]
        if len(otherm)>0:name+=str(len(otherm))
        torch.save(self,f'Models/{name}.model')


# model = FCN(layers,Loss,scheduler='StepLR',argsoptim={'amsgrad':True},argschedu={'step_size':30,'gamma':0.1})
# model.Train(100,x_BC,y_BC,x_PDE)
# optimizer = torch.optim.Adam(model.parameters(),lr=lr,amsgrad=False)
# print(optimizer)
# lmbda = lambda epoch: 0.65 ** epoch
# scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

# print(scheduler.__module__ == 'torch.optim.lr_scheduler')

#! Generate Data

if __name__ == '__main__':
    

    # get the analytical solution over the full domain
    x = torch.linspace(min,max,total_points).view(-1,1) #prepare to NN
    y = f_real(x)
    # print(x.shape, y.shape)

    # fig, ax1 = plt.subplots()
    # ax1.plot(x.detach().numpy(),y.detach().numpy(),color='blue',label='Real Function')
    # #ax1.plot(x_train.detach().numpy(),yh.detach().numpy(),color='red',label='Pred_Train')
    # ax1.set_xlabel('x',color='black')
    # ax1.set_ylabel('f(x)',color='black')
    # ax1.tick_params(axis='y', color='black')
    # ax1.legend(loc = 'upper left')
    # plt.draw();plt.pause(10);plt.close()

    #def get_training_data(x):
    #Nu: Number of training point, # Nf: Number of colloction points
    # Set Boundary conditions x=min & x= max
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
    # #Create Model
    # model = FCN(layers,Loss,lr=0.001,argsoptim={'amsgrad':True})#,scheduler='ReduceLROnPlateau',argschedu={'patience':200,'verbose':True,'factor':0.5})

    # # print(model)
    optimizers = ['Adam', 'SGD', 'Adadelta', 'Adagrad']
    colors = ['b','r','g','y']
    fig,axes = plt.subplots(2,figsize=(16,10))
    # if exists('OtherResults/OptimizerTests.pkl'):
    #     with open('OtherResults/OptimizerTests.pkl','rb') as file:
    #         CompResults=pkl.load(file)
    # else:CompResults={}
    CompResults={}
    for i,optimizer in enumerate(optimizers):
        torch.manual_seed(123)
        print(f'Considering optimizer: {optimizer}\n')
        # if not exists(f'Models/{optimizer}.model'):
        #     t0 = time()
        #     model = FCN(layers,Loss,lr=0.001,optimizer=optimizer)
        #     model.Train(x_BC,y_BC,x_PDE,10000)
        #     model.save(optimizer)
        #     CompResults[optimizer] = {'minLoss':np.min(model.loss_history),'duration':time()-t0}
        #     if exists(f'OtherResults/OptimizerTests.pkl'):
        #         remove('OtherResults/OptimizerTests.pkl')
        #     with open('OtherResults/OptimizerTests.pkl','wb') as file:  
        #         pkl.dump(CompResults,file) 
        #     print('\n',f'Reached minimum loss: {CompResults[optimizer]["minLoss"]:.3e}','\n',f'Time spent computing: {CompResults[optimizer]["duration"]:.3f} s','\n\n')
        # else:
        #     model = torch.load(f'Models/{optimizer}.model')
        Y = []
        History = []
        CompResults[optimizer] = {'minLoss':[],'duration':[]}
        for _ in range(100):
            model = FCN(layers,Loss,lr=0.001,optimizer=optimizer)
            t0 = time()
            model.Train(x_BC,y_BC,x_PDE,10000)
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
        axes[0].fill_between(x.detach().numpy().reshape((-1)),yh-dyh,yh+dyh,color=colors[i],alpha=.3)
        axes[1].fill_between(np.arange(10000)+1,mHistory-dHistory,mHistory+dHistory,color=colors[i],alpha=.3)
        axes[0].plot(x.detach().numpy().reshape((-1)),yh,color=colors[i],ls='-',label=optimizer)
        axes[1].plot(mHistory,color=colors[i],ls='-',label=optimizer)
        del Y, History, dyh, dHistory, mHistory

        
    axes[0].plot(x,f_real(x).detach().numpy(),color='k',label='Real u(x)')
    axes[0].legend(frameon=False,);axes[0].set_xlabel('x',color='black');axes[0].set_ylabel('f(x)',color='black')
    axes[1].set_xlabel('Training Epochs');axes[1].set_ylabel('loss');axes[1].set_yscale('log')
    fig.savefig('Figures/OptimizerTestsLoss.pdf',bbox_inches='tight')
    fig.savefig('Figures/OptimizerTestsLoss.png',bbox_inches='tight')
    plt.draw();plt.pause(10);plt.close(fig)
    CompResults = pd.DataFrame(CompResults).T
    fig2, axes = plt.subplots(1,2,figsize=(16,10))
    bar1=axes[0].bar(CompResults.index,CompResults['mminLoss'],yerr=CompResults['dminLoss'],width=0.4,color=colors,ecolor='black',capsize=3)
    bar2=axes[1].bar(CompResults.index,CompResults['mduration'],yerr=CompResults['dduration'],width=0.4,color=colors,ecolor='black',capsize=3)
    axes[0].set_ylabel('min(loss)'),axes[0].set_yscale('log');axes[1].set_ylabel('simulation time (s)')
    fig2.savefig('Figures/OptimizerTestsTimeMin.pdf',bbox_inches='tight')
    fig2.savefig('Figures/OptimizerTestsTimeMin.png',bbox_inches='tight')
    plt.draw();plt.pause(10);plt.close(fig2)
    # CompResults.plot.bar(subplots=True,rot=90);plt.show()

    # # model.to(device)
    # # params = list(model.parameters())
    # # optimizer = torch.optim.Adam(model.parameters(),lr=lr,amsgrad=False)
    # # for i in tqdm(range(steps)):
    # #     yh = model(x_PDE)
    # #     loss = model.loss(x_BC,y_BC,x_PDE)# use mean squared error
    # #     optimizer.zero_grad()
    # #     loss.backward()
    # #     optimizer.step()
    # #     if i%(steps/10)==0:
    # #         print(loss)
    # # Function
    # yh=model(x.to(model.device))
    # y=f_real(x)
    # #Error
    # print(lossBC(model,x.to(model.device),f_real(x).to(model.device)))
    # # Derivative
    # g=x.to(model.device)
    # g=g.clone()
    # g.requires_grad=True #Enable differentiation
    # f=model(g)
    # f_x=autograd.grad(f,g,torch.ones([g.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0]



    # # Detach from GPU
    # y_plot=y.detach().numpy()
    # yh_plot=yh.detach().cpu().numpy()
    # f_x_plot=f_x.detach().cpu().numpy()



    # # Plot
    # fig, ax1 = plt.subplots()
    # ax1.plot(x,y_plot,color='blue',label='Real u(x)')
    # ax1.plot(x,yh_plot,color='red',label='Predicted u(x)')
    # ax1.plot(x,f_x_plot,color='green',label='Computed u\'(x)')
    # ax1.set_xlabel('x',color='black')
    # ax1.set_ylabel('f(x)',color='black')
    # ax1.tick_params(axis='y', color='black')
    # ax1.legend(loc = 'upper left')

    # plt.show()
