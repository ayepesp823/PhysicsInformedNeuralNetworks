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
steps = 10000
lr = 0.01
layers = np.array([2,150,150,150,1])#50,50,20,50,50
xmin, xmax = 0,1
ymin, ymax = 0,1
total_points_x = 100
total_points_y = 100
Nu, Nf = 500, 10000#? #Nu: number of training points of boundary #Nf: number of collocation points (to evaluate PDE in)

k0 = 2*np.pi*2
def f_real(x,t):
    return torch.sin(k0 * x) * torch.sin(k0 * t)
'Loss Function'
def ResPDE(Xpde,Fcomp):
    f_x_y = autograd.grad(Fcomp,Xpde,torch.ones([Xpde.shape[0], 1]).to(model1.device), retain_graph=True, create_graph=True)[0] #first derivative
    f_xx_yy = autograd.grad(f_x_y,Xpde,torch.ones(Xpde.shape).to(model1.device), create_graph=True)[0] #second derivative
    
    f_xx=f_xx_yy[:,[0]]# we select the 1st element for x (the second one is t) (Remember the input X=[x,t]) 
    f_yy=f_xx_yy[:,[1]]# we select the 1st element for x (the second one is t) (Remember the input X=[x,t]) 
    f=-f_yy - f_xx - (k0**2)*Fcomp - (k0**2)*torch.sin(k0 * Xpde[:, 0:1]) * torch.sin(k0 * Xpde[:, 1:2])
    return [f]

def transform(x, y):
    res = x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
    return res * y

'Model'
model1 = torch.load()
model1.load_state_dict(torch.load(f'Models/{model1.name}/Best/StateDict_{model1.n}.model'))
model2 = torch.load()
model2.load_state_dict(torch.load(f'Models/{model2.name}/Best/StateDict_{model2.n}.model'))

'Gen Data'
x=torch.linspace(xmin,xmax,total_points_x).view(-1,1)
t=torch.linspace(ymin,ymax,total_points_y).view(-1,1)
# Create the mesh 
X,Y=torch.meshgrid(x.squeeze(1),t.squeeze(1))
# Evaluate real function
u_real=f_real(X,Y)

# Transform the mesh into a 2-column vector
x_test=torch.hstack((X.transpose(1,0).flatten()[:,None],Y.transpose(1,0).flatten()[:,None]))
u_test=u_real.transpose(1,0).flatten()[:,None] # Colum major Flatten (so we transpose it)

def FCDT(Xcdt,Ucdt):
    uborder = torch.zeros(Xcdt.shape[0],1)
    return model2.criterion(Ucdt,uborder)

fig = plt.figure(figsize=(16,30))

axr = fig.add_axes([0.25,0.775,0.4,0.2])

axp1 = fig.add_axes([0.08,0.5,0.4,0.2])
axp2 = fig.add_axes([0.52,0.5,0.4,0.2])

axe1 = fig.add_axes([0.08,0.275,0.4,0.2])
axe2 = fig.add_axes([0.52,0.275,0.4,0.2])

for ax in (axr,axp1,axp2,axe1,axe2):
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
axp1.get_shared_x_axes().join(axp1,axe1)
axp2.get_shared_x_axes().join(axp2,axe2)
axp2.get_shared_y_axes().join(axp2,axp1)
axe2.get_shared_y_axes().join(axe2,axe1)

axp1.set_xticklabels([])
axp2.set_xticklabels([])
axp2.set_yticklabels([])
axe2.set_yticklabels([])

axl = fig.add_axes([0.08,0.05,0.8,0.2])

axl.set_xlim(0,10000)
axl.set_yscale('log');axl.set_ylabel('loss');axl.set_xlabel('Epochs')

plt.show()