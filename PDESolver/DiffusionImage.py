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

x=torch.linspace(-1,1,200).view(-1,1)
t=torch.linspace( 0,1,100).view(-1,1)
# Create the mesh 
X,T=torch.meshgrid(x.squeeze(1),t.squeeze(1))
def f_real(x,t):
    return torch.exp(-t)*(torch.sin(np.pi*x))#torch.exp(-t)*(torch.sin(x)+(torch.sin(2*x))/2+(torch.sin(3*x))/3+(torch.sin(4*x))/4+(torch.sin(8*x))/8)
y_real=f_real(X,T)
# Transform the mesh into a 2-column vector
x_test=torch.hstack((X.transpose(1,0).flatten()[:,None],T.transpose(1,0).flatten()[:,None]))
y_test=y_real.transpose(1,0).flatten()[:,None] # Colum major Flatten (so we transpose it)



norm1,mapper1 = ColorMapMaker(y_test.min(),y_test.max(),'cool')


fig = plt.figure(figsize=(16,30))

axr1 = fig.add_axes([0.08,0.775,0.4,0.2])
axr2 = fig.add_axes([0.52,0.775,0.4,0.2])

axp1 = fig.add_axes([0.08,0.5,0.4,0.2])
axp2 = fig.add_axes([0.52,0.5,0.4,0.2])

axe1 = fig.add_axes([0.08,0.275,0.4,0.2])
axe2 = fig.add_axes([0.52,0.275,0.4,0.2])

for ax in (axr1,axp1,axe1):
    ax.set_xlim(0,1)
    ax.set_ylim(-1,1)
for ax in (axr2,axp2,axe2):
    ax.set_xlim(0,1)
    ax.set_ylim(-np.pi,np.pi)

axp1.get_shared_x_axes().join(axp1,axe1)
axp2.get_shared_x_axes().join(axp2,axe2)

axp1.set_xticklabels([])
axp2.set_xticklabels([])
axp2.set_yticklabels([])
axe2.set_yticklabels([])

axl = fig.add_axes([0.08,0.05,0.8,0.2])

axl.set_xlim(0,10000)
axl.set_yscale('log');axl.set_ylabel('loss');axl.set_xlabel('Epochs')

plt.show()