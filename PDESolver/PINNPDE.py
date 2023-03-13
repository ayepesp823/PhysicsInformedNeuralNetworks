"""Script inspired by https://github.com/jdtoscano94/Learning-Python-Physics-Informed-Machine-Learning-PINNs-DeepONets repo"""

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
import warnings
from os.path import exists
from os import makedirs, listdir, remove
warnings.filterwarnings("ignore")
try:plt.style.use('../figstyle.mplstyle')
except:pass

#! Tunning parameters


#! Neural Network

class FCN(nn.Module):
    #?Neural Network
    def __init__(self,layers,ResPDE:callable,HardConstrain=None,activation=nn.Tanh(),lr=0.5,criterion=nn.MSELoss(reduction ='mean'),
                    optimizer='Adam',scheduler=None,argsoptim={},argschedu={}):
        super().__init__() #? call __init__ from parent class 

        'activation function'
        self.activation = activation#? activation function applied after each layer

        'criterion'#? used for computing the loss out of the residuals estimated
        self.criterion = criterion#? for forward(input) and predicted output

        'hard constrain'#? used for imposing boundary conditions
        if HardConstrain is not None and callable(HardConstrain):
            self.HardConstrain = HardConstrain
        else:
            self.HardConstrain = None
            print(f'Argument HardConstrain set to type: {type(HardConstrain)}, expected type function')
        
        'Function to compute residuals of the PDE'
        self.ResPDE = ResPDE

        'Initialise neural network as a list using nn.Modulelist'  
        self.layers = layers
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])#? construction of the network
        self.iter = 0

        'device to be used for training'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        'Xavier Normal Initialization'
        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(self.layers)-1):
            
            # weights from a normal distribution with 
            # Recommended gain value for tanh = 5/3?
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)   
        self.to(self.device)

        'optimizer selection'
        try:
            self.optimizer = getattr(torch.optim, optimizer)(self.parameters(), lr=lr, **argsoptim)
        except:
            self.optimizer = torch.optim.Adam(self.parameters(),lr=lr,**argsoptim)
            print(f'\nOptimizer "{optimizer}" was not found, Adam was used instead\n')
        
        'learning rate scheduler selections'#? if wanted, the code can implement a learning rate scheduler
        try:
            self.scheduler = getattr(torch.optim.lr_scheduler,scheduler)
        except:
            self.scheduler = None
            if scheduler is not None:print(f'Learning Rate Scheduler "{scheduler}" was not found. Ingoring option.')
        if self.scheduler is not None:self.scheduler = self.scheduler(self.optimizer,**argschedu)
    
    'foward pass'
    def forward(self,x):
        'this function computes the output of the neural network'
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                
        a = x.float()
        for i in range(len(self.layers)-2):  
            z = self.linears[i](a)              
            a = self.activation(z)    
        a = self.linears[-1](a)
        if self.HardConstrain is not None:
            a = self.HardConstrain(x,a)#? optional step to impose boundary conditions
        return a

    def lossBD(self,Xbd,Ybd):
        """
        This function computed the loss generated due to boundary conditions

        Args:
            Xbd (_type_): collocation points in the boundary.
            Ybd (_type_): expected output for boundary points.

        Returns:
            lossBD (_type_): loss due to the boundary collocation points output estimation.
        """
        return self.criterion(self.forward(Xbd),Ybd)

    def lossPDE(self,Xpde):
        """
        This function computes the loss generated due to collocation points,
        (aka points inside the regimen for computing the PDE). It used the
        residuals of the PDE so it is asummed for the exact solution f that:
        PDE(f,x)=0

        Args:
            Xpde (_type_): Collocation points

        Returns:
            lossPDE (_type_): loss due to the collocation points output estimation.
        """
        XpdeC = Xpde.clone()
        XpdeC.requires_grad = True
        f_hat = torch.zeros(Xpde.shape[0],1).to(self.device)
        f_est = self.ResPDE(XpdeC,self.forward(XpdeC))
        return self.criterion(f_est,f_hat)

    def loss(self,Xpde,Xbd=None,Ybd=None,weight=1.):
        """
        this function computes the total loss for the PDE solution estimation problem,
        it takes into account the loss generated by

        Args:
            Xpde (_type_): collocation points.
            Xbd (torch.tensor, optional): Points in boundary. Defaults to None.
            Ybd (torch.tensor, optional): Expected output for boundary. Defaults to None.
            weight (float, optional): _description_. Defaults to 1.

        Returns:
            total_loss (_type_): lossPDE or weight*lossBD + lossPDE.
        """
        total_loss = self.lossPDE(Xpde)
        if self.HardConstrain is None and Xbd is not None:
            total_loss += weight*self.lossBD(Xbd,Ybd)
        return total_loss
    

    def Train(self,Xpde,Xbd=None,Ybd=None,weight=1,nEpochs=100,BatchSize=None,history=True,Verbose=False,nLoss=1000,pScheduler=False,
                Test=[None,None],nTest=10,PH=True):
        """_summary_

        Args:
            Xpde (_type_): collocation points.
            Xbd (torch.tensor, optional): Points in boundary. Defaults to None.
            Ybd (torch.tensor, optional): Expected output for boundary. Defaults to None.
            weight (float, optional): _description_. Defaults to 1.
            nEpochs (int, optional): number of epochs to train. Defaults to 100.
            BatchSize (_type_, optional): _description_. Defaults to None.
            history (bool, optional): compute history of the training. Defaults to True.
            Verbose (bool, optional): show training history in prompt. Defaults to False.
            nLoss (int, optional): how many iterations to wait for printing loss in prompt. Defaults to 1000.
            pScheduler (bool, optional): store scheduler history for later plotting. Defaults to False.
            Test (list, optional): [x_test,y_test] values to compute loss relative exact solutions (if known). Defaults to [None,None].
            nTest (int, optional): how many iterations to wait for computing test loss. Defaults to 10.
            PH (bool, optional): plot loss history. Defaults to True.

        Returns:
            _type_: _description_
        """
        if self.HardConstrain is None and Xbd is None:
            print('No boundary conditions implemented.')
        if Xbd is not None and Ybd is None:
            print('Asuming boundary condition set to 0.')
            Ybd = torch.zeros(Xbd.shape[0],1).to(self.device)
        if history:
            self.total_loss_history = []
            self.pde_loss_history = []
            self.bd_loss_history = []
            if Test[0] is not None and Test[1] is not None:
                self.test_history_loss = []
                self.test_history_iter = []
        if self.scheduler is not None: 
            self.schhist=None
        iterator = range(nEpochs)
        if Verbose: iterator = tqdm(iterator, bar_format='{desc}: {percentage:6.2f}% |{bar}{r_bar}')
        for n in iterator:
            self.optimizer.zero_grad()
            Loss = self.loss(Xpde,Xbd,Ybd,weight)
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
            if history:
                self.total_loss_history.append(Loss.detach().cpu().numpy())
                if self.HardConstrain is None and Xbd is not None:
                    self.pde_loss_history.append(self.lossPDE(Xpde).detach().cpu().numpy())
                    self.bd_loss_history.append(weight*self.lossBD(Xbd,Ybd).detach().cpu().numpy())

                if Test[0] is not None and Test[1] is not None and n%nTest==0:
                    # try:
                        self.test_history_iter.append(n)
                        self.test_history_loss.append(self.criterion(self.forward(Test[0]),Test[1]).detach().cpu().numpy())
        if PH:
            return self.PlotHistory(pScheduler)
                    # except:pass

    def PlotHistory(self,pScheduler=False):
        fig, ax = plt.subplots(1,figsize=(16,10))
        if pScheduler:
            # try:
                lrp = ax.plot(self.schhist,color='k',ls=':',alpha=0.7)
                legend1 = plt.legend(lrp,['Learning rate'],loc='lower left')
                ax.add_artist(legend1)
                legend1.get_frame().set_alpha(None)
                legend1.get_frame().set_facecolor((1, 0, 0, 0.1))
            # except:pass
        
        total = ax.plot(self.total_loss_history,color='m',ls='-',label='Total')
        legends = total
        if len(self.pde_loss_history)>0:
            pde = ax.plot(self.pde_loss_history,color='b',ls='-',label='PDE')
            bd = ax.plot(self.bd_loss_history,color='r',ls='-',label='BD')
            legends+=pde;legends+=bd
        try:test = ax.plot(self.test_history_iter,self.test_history_loss,color='g',ls='--',label='Test');legends+=test
        except:pass
        labs = [curve.get_label() for curve in legends]
        legend2 = plt.legend(legends,labs,title='Loss Type:',markerfirst=False,edgecolor='k',alignment='right',loc='upper right',title_fontsize=12)

        legend2.get_frame().set_alpha(None)
        legend2.get_frame().set_facecolor((0, 0, 0, 0.0))
        ax.add_artist(legend2)
        ax.set_yscale('log');ax.set_ylabel('loss');ax.set_xlabel('Epochs')
        return fig, ax

    def save(self,name):
        if not exists('Models'):
            makedirs('Models')
        print(name)
        otherm = [file.replace(name,'').replace('.model','') for file in listdir('Models') if file.startswith(name)]
        if len(otherm)>0:name+=str(len(otherm))
        torch.save(self,f'Models/{name}.model')


#! Generate Data

if __name__ == '__main__':
    'Complementary function'
    def plot3D_Matrix(x,t,y):
        X,T= x,t
        F_xt = y
        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(T,X, F_xt,20,cmap="rainbow")
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('F(x,t)')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        plt.show()
        ax = plt.axes(projection='3d')
        ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(),cmap="rainbow")
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_zlabel('f(x,t)')
        plt.show()

    'Initial set up'
    torch.manual_seed(1234)
    np.random.seed(1234)
    steps = 20000
    lr = 1e-3
    layers = np.array([2,32,1])
    xmin, xmax = -1,1
    tmin, tmax = 0,1
    total_points_x = 200
    total_points_t = 100
    Nu, Nf = 100, 10000#? #Nu: number of training points of boundary #Nf: number of collocation points (to evaluate PDE in)
    def f_real(x,t):
        return torch.exp(-t)*(torch.sin(np.pi*x))
    'Loss Functions'
    def PDE(x):
        return torch.exp(-x[:, 1:])* (torch.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * torch.sin(np.pi * x[:, 0:1]))
    #Loss BC
    def lossBC(model,x_BC,y_BC):
        loss_BC=model.criterion(model.forward(x_BC),y_BC)
        return loss_BC
    #Loss PDE
    def lossPDE(model,x_PDE,PDE):
        g=x_PDE.clone()
        f_hat = torch.zeros(x_PDE.shape[0],1).to(model.device)
        g.requires_grad=True #Enable differentiation
        f=model.forward(g)
        f_x_t = autograd.grad(f,g,torch.ones([g.shape[0], 1]).to(model.device), retain_graph=True, create_graph=True)[0] #first derivative
        f_xx_tt = autograd.grad(f_x_t,g,torch.ones(g.shape).to(model.device), create_graph=True)[0] #second derivative

        f_t=f_x_t[:,[1]]# we select the 2nd element for t (the first one is x) (Remember the input X=[x,t]) 
        f_xx=f_xx_tt[:,[0]]# we select the 1st element for x (the second one is t) (Remember the input X=[x,t]) 
        f=f_t-f_xx+ PDE(g)
        return model.criterion(f,f_hat)

    def loss(model,x_BC,y_BC,x_PDE,lossBC,lossPDE,PDE):
        loss_bc=lossBC(model,x_BC,y_BC)
        loss_pde=lossPDE(model,x_PDE,PDE)
        return loss_bc+loss_pde
    
    'Model'
    model = FCN(layers,loss,lr=lr)

    'Gen Data'
    x=torch.linspace(xmin,xmax,total_points_x).view(-1,1)
    t=torch.linspace(tmin,tmax,total_points_t).view(-1,1)
    # Create the mesh 
    X,T=torch.meshgrid(x.squeeze(1),t.squeeze(1))
    # Evaluate real function
    y_real=f_real(X,T)
    # Transform the mesh into a 2-column vector
    x_test=torch.hstack((X.transpose(1,0).flatten()[:,None],T.transpose(1,0).flatten()[:,None]))
    y_test=y_real.transpose(1,0).flatten()[:,None] # Colum major Flatten (so we transpose it)
    # Domain bounds
    lb=x_test[0] #first value
    ub=x_test[-1] #last value 
    #Initial Condition
    #Left Edge: x(x,0)=sin(x)->xmin=<x=<xmax; t=0
    left_X=torch.hstack((X[:,0][:,None],T[:,0][:,None])) # First column # The [:,None] is to give it the right dimension
    left_Y=torch.sin(np.pi*left_X[:,0]).unsqueeze(1)
    #Boundary Conditions
    #Bottom Edge: x=min; tmin=<t=<max
    bottom_X=torch.hstack((X[0,:][:,None],T[0,:][:,None])) # First row # The [:,None] is to give it the right dimension
    bottom_Y=torch.zeros(bottom_X.shape[0],1)
    #Top Edge: x=max; 0=<t=<1
    top_X=torch.hstack((X[-1,:][:,None],T[-1,:][:,None])) # Last row # The [:,None] is to give it the right dimension
    top_Y=torch.zeros(top_X.shape[0],1)
    #Get all the training data into the same dataset
    X_train=torch.vstack([left_X,bottom_X,top_X])
    Y_train=torch.vstack([left_Y,bottom_Y,top_Y])
    #Choose(Nu) points of our available training data:
    idx=np.random.choice(X_train.shape[0],Nu,replace=False)
    X_train_Nu=X_train[idx,:].float().to(model.device)
    Y_train_Nu=Y_train[idx,:].float().to(model.device)
    # Collocation Points (Evaluate our PDe)
    #Choose(Nf) points(Latin hypercube)
    X_train_Nf=lb+(ub-lb)*lhs(2,Nf) # 2 as the inputs are x and t
    X_train_Nf=torch.vstack((X_train_Nf,X_train_Nu)).float().to(model.device) #Add the training poinst to the collocation points
    X_test=x_test.float().to(model.device) # the input dataset (complete)
    Y_test=y_test.float().to(model.device)
    # y = torch.zeros_like(X_train_Nf.T)
    # index = torch.LongTensor([1,0])
    # y[index] = X_train_Nf.T
    # y = y.T
    # print(X_train_Nf,'\n',y)
    model.Train([X_train_Nu,Y_train_Nu,X_train_Nf,lossBC,lossPDE,PDE],Verbose=True,nEpochs=steps)
    fig, ax = plt.subplots(1,figsize=(16,10))
    ax.plot(model.loss_history,ls='-',color='b')
    ax.set_yscale('log');ax.set_ylabel('loss');ax.set_xlabel('Epochs')
    fig.savefig('DifussionEquationLoss.png',bbox_inches='tight')
    plt.show()
    y1=model(X_test)
    x1=X_test[:,0]
    t1=X_test[:,1]
    arr_x1=x1.reshape(shape=[100,200]).transpose(1,0).detach().cpu()
    arr_T1=t1.reshape(shape=[100,200]).transpose(1,0).detach().cpu()
    arr_y1=y1.reshape(shape=[100,200]).transpose(1,0).detach().cpu()
    arr_y_test=y_test.reshape(shape=[100,200]).transpose(1,0).detach().cpu()

    plot3D_Matrix(arr_x1,arr_T1,arr_y1)




