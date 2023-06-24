"""Script inspired by https://github.com/jdtoscano94/Learning-Python-Physics-Informed-Machine-Learning-PINNs-DeepONets repo"""

import torch
from torch import nn

from matplotlib import pyplot as plt

import numpy as np
from tqdm import tqdm
from time import time
import warnings
from os.path import exists
from os import makedirs, listdir
warnings.filterwarnings("ignore")
try:plt.style.use('../figstyle.mplstyle')
except:pass

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


#! Neural Network

class FCN(nn.Module):
    #?Fully Connected Neural Network
    def __init__(self,layers,ResPDE:callable,HardConstrain=None,activation=nn.Tanh(),lr=0.5,criterion=nn.MSELoss(reduction ='mean'),
                    optimizer='Adam',scheduler=None,argsoptim={},argschedu={},numequ=1,numcdt=0,focusmin=0.05,focusmax=0.05,name='PDESolver'):
        super().__init__() #? call __init__ from parent class 
        'Initialization'
        if not exists('Models'):
            makedirs('Models')
        if not exists('Figures'):
            makedirs('Figures')
        'model name'
        self.name=name
        if not exists(f'Models/{self.name}'):
            makedirs(f'Models/{self.name}')
            makedirs(f'Models/{self.name}/Final')
            makedirs(f'Models/{self.name}/Best')
        if not exists(f'Figures/{self.name}'):
            makedirs(f'Figures/{self.name}')
            makedirs(f'Figures/{self.name}/Images')
            makedirs(f'Figures/{self.name}/Instances')
        self.n = len([file for file in listdir(f'Models/{self.name}/Final')])+1

        'activation function'
        self.activation = activation#? activation function applied after each layer

        'criterion'#? used for computing the loss out of the residuals estimated
        self.criterion = criterion#? for forward(input) and predicted output

        'hard constrain'#? used for imposing boundary conditions
        if callable(HardConstrain):
            self.HardConstrain = HardConstrain
        else:
            self.HardConstrain = None
            if HardConstrain is not None:
                print(f'Argument HardConstrain set to type: {type(HardConstrain)}, expected type function')

        'Number of equations in the system'
        self.numequ = numequ
        self.numcdt  = numcdt
        
        'zoom to loss plots'

        try:self.focusmin = float(focusmin)
        except:self.focusmin = None
        try:self.focusmax = float(focusmax)
        except:self.focusmax = None

        'Function to compute residuals of the PDE'
        self.ResPDE = ResPDE#! OUTPUT MOST BE ITERABLE

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
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=5/3)
            
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)   
        self.to(self.device)

        'Initialize best loss'
        self.best_valid_loss = float('inf');self.best_loss_it = 0

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

    def lossCDT(self):
        """
        This function computed the loss generated due to boundary conditions

        Args:
            Xbd (_type_): collocation points in the boundary.
            Ybd (_type_): expected output for boundary points.

        Returns:
            lossBD (_type_): loss due to the boundary collocation points output estimation.
        """
        loss_bd = []
        for k in range(self.numcdt):
            Xcdt, FunCDT = self.CDT[k][:2]
            XcdtC = Xcdt.clone()
            XcdtC.requires_grad = True
            try:weight = torch.tensor(float(self.CDT[k][2]))
            except:weight = torch.tensor(1.)
            loss_bd.append(weight*FunCDT(XcdtC,self.forward(XcdtC)))
        return loss_bd

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
        f_hat = torch.zeros(Xpde.shape[0],1).to(self.device)
        XpdeC = Xpde.clone()
        XpdeC.requires_grad = True
        loss_pde = []
        Ests = self.ResPDE(XpdeC,self.forward(XpdeC))
        for f_est in Ests:
            loss_pde.append(self.criterion(f_est,f_hat))
        return loss_pde

    def loss(self):
        """
        this function computes the total loss for the PDE solution estimation problem,
        it takes into account the loss generated by

        Returns:
            total_loss (_type_): lossPDE or weight*lossBD + lossPDE.
        """
        total_loss = sum(self.lossPDE(self.train_Xpde))
        if self.numcdt>0:
            total_loss+=sum(self.lossCDT())
        return total_loss
    
    def closure(self):
        self.optimizer.zero_grad()
        Loss = self.loss()
        Loss.backward()
        return Loss

    def Train(self,Xpde,CDT=None,nEpochs=100,BatchSize=None,history=True,Verbose=False,nLoss=1000,pScheduler=False,
                Test=[None,None],nTest=10,PH=True,SH=False,SB=True,RE=True):
        """_summary_

        Args:
            Xpde (_type_): collocation points.
            Xbd (torch.tensor, optional): Points in boundary. Defaults to None.
            Ybds (list of torch.tensor, optional): Expected output for boundary. Defaults to None.
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
            SH (bool, optional): save detailed history. Defaults to False.
            SB (bool, optional): save best prediction. Defaults to True.
            RE (bool, optional): rapid exis. Defaults to True
            CDT (tuple(tuple(Xcdt,RedBD,weight{optional}))): conditions to be applied, must coincide with numbd. Defaults to None.

        Returns:
            _type_: _description_
        """
        if nEpochs<100:RE=False
        if nEpochs<= 30000:maxepochs=0.05*nEpochs
        else:maxepochs = 1500
        if self.HardConstrain is None and CDT is None:
            print('No conditions implemented.')
        if CDT is not None:
            if len(CDT) != self.numcdt:
                print(CDT)
                raise TypeError('Number of conditions not matching, killing training process')
        self.CDT = CDT
        self.train_Xpde = Xpde.clone()
        self.loss_history = {'total_loss':[],'total_pde_loss':[],'total_bd_loss':[]}
        SH = SH and (self.numequ>1 or self.numcdt>0)
        if SH:
            for i in range(self.numequ):
                self.loss_history[f'pde_loss_{i}'] = []
            for i in range(self.numcdt):
                self.loss_history[f'bd_loss_{i}'] = []
        if Test[0] is not None and Test[1] is not None:
            self.loss_history['test_loss'] = []
            self.loss_history['test_iter'] = []
                
        if self.scheduler is not None: 
            self.schhist=None
        iterator = range(nEpochs)
        if Verbose: iterator = tqdm(iterator, bar_format='{desc}: {percentage:6.2f}% |{bar}{r_bar}')
        t0 = time()
        for n in iterator:
            try:
                self.optimizer.step(self.closure)
                Loss = self.loss()
            except:
                Loss = self.closure()
                self.optimizer.step()
            if SB and Loss.detach().cpu().numpy()<self.best_valid_loss and Loss.detach().cpu().numpy()>1e-15:
                self.best_valid_loss = Loss.detach().cpu().numpy()
                self.best_loss_it = n+1
                # if Verbose and n>2*maxepochs:
                #     print(f"\nBest validation loss: {self.best_valid_loss:.3e}")
                #     print(f"\nSaving best model for epoch: {n+1}\n")
                torch.save(self.state_dict(),f'Models/{self.name}/Best/StateDict_{self.n}.model')

            if self.scheduler is not None: 
                if pScheduler:
                    if self.schhist is not None:self.schhist.append(self.optimizer.param_groups[0]["lr"])
                    else:self.schhist = [self.optimizer.param_groups[0]["lr"]]
                try:self.scheduler.step()
                except:self.scheduler.step(Loss)
            if Verbose and n%nLoss==0:
                print(f'\nloss at iteration {n}: {Loss:.3e}\nCurrent best loss: {self.best_valid_loss:.3E}')
            self.loss_history['total_loss'].append(Loss.detach().cpu().numpy())
            if self.numcdt>0:
                loss_pde = self.lossPDE(self.train_Xpde)#.detach().cpu().numpy()
                loss_bd = self.lossCDT()
                self.loss_history['total_pde_loss'].append(0)
                for respde in loss_pde:
                    self.loss_history['total_pde_loss'][-1] += respde.item()
                self.loss_history['total_bd_loss'].append(0)
                for bd in loss_bd:
                    if bd is not None:
                        self.loss_history['total_bd_loss'][-1] += bd.item()
                if SH:
                    for i in range(self.numequ):
                        self.loss_history[f'pde_loss_{i}'].append(loss_pde[i].item())
                    for i in range(self.numcdt):
                        if loss_bd[i] is not None:self.loss_history[f'bd_loss_{i}'].append(loss_bd[i].item())
            else:
                if SH:
                    loss_pde = self.lossPDE(self.train_Xpde)
                    for i in range(self.numequ):
                        self.loss_history[f'pde_loss_{i}'].append(loss_pde[i].item())

            if Test[0] is not None and Test[1] is not None and n%nTest==0:
                # try:
                    self.loss_history['test_iter'].append(n)
                    self.loss_history['test_loss'].append(self.criterion(self.forward(Test[0]),Test[1]).detach().cpu().numpy())
            
            if n>0.2*nEpochs and RE and ((Loss<1e-4 and np.std(self.loss_history['total_loss'][-int(maxepochs):])/np.mean(self.loss_history['total_loss'][-int(maxepochs):])<1e-3) or np.std(self.loss_history['total_loss'][-int(1.5*maxepochs):])/np.mean(self.loss_history['total_loss'][-int(1.5*maxepochs):])<1e-4):
                print(f'Model already converged at epoch: {n}, breaking trainning.')
                print(f'Last Loss: {Loss:.3e}')
                print(f'Std of last {int(maxepochs)} Loss: {np.std(self.loss_history["total_loss"][-int(maxepochs):]):.3e}')
                print(f'Std of last {int(1.5*maxepochs)} Loss: {np.std(self.loss_history["total_loss"][-int(1.5*maxepochs):]):.3e}')
                break
        self.time = time()-t0
        if PH:
            return self.PlotHistory(pScheduler,SH=SH)
                    # except:pass

    def PlotHistory(self,pScheduler=False,SH=False):
        fig, ax = plt.subplots(1,figsize=(16,10))
        fig.subplots_adjust(hspace=0.01,wspace=0)
        if pScheduler:
            try:
                lrp = ax.plot(self.schhist,color='k',ls='-.-',alpha=0.7)
                legend1 = plt.legend(lrp,['Learning rate'],loc='lower left')
                ax.add_artist(legend1)
                legend1.get_frame().set_alpha(None)
                legend1.get_frame().set_facecolor((1, 0, 0, 0.1))
            except:pass
        
        total = ax.plot(self.loss_history['total_loss'],color='m',ls='-',label='Total')
        legends = total
        if len(self.loss_history['total_pde_loss'])>0:
            pde = ax.plot(self.loss_history['total_pde_loss'],color='b',ls='-',label='PDE')
            bd = ax.plot(self.loss_history['total_bd_loss'],color='r',ls='-',label='BD')
            legends+=pde;legends+=bd
        try:test = ax.plot(self.loss_history['test_iter'],self.loss_history['test_loss'],color='g',ls='--',label='Test');legends+=test
        except:pass
        labs = [curve.get_label() for curve in legends]
        legend2 = plt.legend(legends,labs,title='Loss Type:',markerfirst=False,edgecolor='k',alignment='right',loc='upper right',title_fontsize=12)

        legend2.get_frame().set_alpha(None)
        legend2.get_frame().set_facecolor((1, 1, 1, 1))
        ax.add_artist(legend2)
        ax.set_yscale('log');ax.set_ylabel('loss');ax.set_xlabel('Epochs')
        ymin,ymax=ax.get_ylim()
        try:nymin = min(self.loss_history[f'total_loss'])*(1-self.focusmin)
        except: nymin = ymin
        try:nymax = max(self.loss_history[f'total_loss'])*(1+self.focusmax)
        except: nymax = ymax
        if nymin>ymin:ymin=nymin
        if ymax>nymax:ymax=nymax
        ax.set_ylim(ymin,ymax)
        fig.savefig(f'Figures/{self.name}/Images/Total_loss_{self.n}.png',bbox_inches='tight')
        if SH:
            cmap1 = get_cmap(3*self.numequ+1);cmap2 = get_cmap(3*self.numcdt+1)
            if self.numcdt>1 and self.numequ>1:
                fig2, ax2 = plt.subplots(2,figsize=(16,15),sharex=True)
            else:
                fig2, ax2 = plt.subplots(1,figsize=(16,10))
                ax2 = [ax2]
            fig2.subplots_adjust(hspace=0.01,wspace=0)
            name=''
            a = 0
            if self.numequ>1:
                name = '_pde'
                a = 1
                leg = ax2[0].plot(self.loss_history[f'total{name}_loss'],color='k',ls='-',label='Total')
                for i in range(self.numequ):
                    leg += ax2[0].plot(self.loss_history[f'pde_loss_{i}'],color=cmap1(3*i),ls='-',label=f'Eq {i+1}')
                labs = [curve.get_label() for curve in leg]
                legend2 = ax2[0].legend(leg,labs,title='PDE Loss:',markerfirst=False,edgecolor='k',alignment='right',loc='upper right',title_fontsize=12)
                legend2.get_frame().set_alpha(None)
                legend2.get_frame().set_facecolor((1, 1, 1, 1))
                ax2[0].set_yscale('log');ax2[0].set_ylabel('loss')
                ymin,ymax=ax2[0].get_ylim()
                try:nymin = min(self.loss_history[f'total{name}_loss'])*(1-self.focusmin)
                except: nymin = ymin
                try:nymax = max(self.loss_history[f'total{name}_loss'])*(1+self.focusmax)
                except: nymax = ymax
                if nymin>ymin:ymin=nymin
                if ymax>nymax:ymax=nymax
                ax2[0].set_ylim(ymin,ymax)
            if self.numcdt>1:
                leg = ax2[a].plot(self.loss_history['total_bd_loss'],color='k',ls='-',label='Total')
                for i in range(self.numcdt):
                    if len(self.loss_history[f'bd_loss_{i}'])>0:leg += ax2[a].plot(self.loss_history[f'bd_loss_{i}'],color=cmap2(3*i),ls='-',label=f'Func {i+1}')
                labs = [curve.get_label() for curve in leg]
                legend2 = ax2[a].legend(leg,labs,title='BD Loss:',markerfirst=False,edgecolor='k',alignment='right',loc='upper right',title_fontsize=12)
                legend2.get_frame().set_alpha(None)
                legend2.get_frame().set_facecolor((1, 1, 1, 1))
                ax2[a].set_yscale('log');ax2[a].set_ylabel('loss')
                
                ymin,ymax=ax2[a].get_ylim()
                try:nymin = min(self.loss_history[f'total_bd_loss'])*(1-self.focusmin)
                except: nymin = ymin
                try:nymax = max(self.loss_history[f'total_bd_loss'])*(1+self.focusmax)
                except: nymax = ymax
                if nymin>ymin:ymin=nymin
                if ymax>nymax:ymax=nymax
                ax2[a].set_ylim(ymin,ymax)
            ax2[-1].set_xlabel('Epochs')
            fig2.savefig(f'Figures/{self.name}/Images/Detailed_loss_{self.n}.png',bbox_inches='tight')
            return fig, ax, fig2, ax2
        return fig, ax

    def save(self):
        if not exists('Models'):
            makedirs('Models')
        torch.save(self,f'Models/{self.name}/Final/Trained_{self.n}.model')


#! Generate Data

