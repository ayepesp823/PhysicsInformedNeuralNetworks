import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from time import time
from os import listdir
import warnings
from os.path import exists
from os import makedirs, listdir, remove
from matplotlib import cm
from PIL import Image
import glob

warnings.filterwarnings("ignore")
try:plt.style.use('../figstyle.mplstyle')
except:pass

nx = 40*2+1
ny = 40*2+1
nt = 500
nit = 50
c = 1

dx = 2 / (nx - 1) 
dy = 2 / (ny - 1) 

#Initializing arrays
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X,Y = np.meshgrid(x,y)

rho = 1
nu = .1
dt = 0.001

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))
nt = 400

def build_up_b(b,rho, dt, u , v, dx, dy):
    
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                    (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                    2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                    (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                    ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    
    return b

def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)
    pn = p.copy()
    
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                            (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                            (2 * (dx**2 + dy**2)) -
                            dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                            b[1:-1,1:-1])

        p[:, -1] = p[:, -2] ##dp/dy = 0 at x = 2
        p[0, :] = p[1, :]  ##dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]    ##dp/dx = 0 at x = 0
        p[-1, :] = 0        ##p = 0 at y = 2
        
    return p

def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))
    Us = [];Vs = [];Ps = []
    for n in tqdm(range(nt)):
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                        un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                        dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                        nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                        dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                        (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                        (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                        (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                        (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1    #set velocity on cavity lid equal to 1
        v[0, :] = 0
        v[-1, :]=0
        v[:, 0] = 0
        v[:, -1] = 0
        
        Us.append(u.copy());Vs.append(v.copy());Ps.append(p.copy())
    return Us,Vs,Ps

if not exists('results.pkl'):
    Us,Vs,Ps = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)
    with open('results.pkl','wb') as file:
        pkl.dump([Us,Vs,Ps],file)
    
    
else:
    with open('results.pkl','rb') as file:
        Us,Vs,Ps = pkl.load(file)

umin = min([u.min() for u in Us])
umax = max([u.max() for u in Us])
vmin = min([v.min() for v in Vs])
vmax = max([v.max() for v in Vs])
levels = np.linspace(-4,4,81)
Titles = [r'$u$ (m/s)',r'$v$ (m/s)',r'$P$ (Pa)']
# for t in tqdm(range(nt)):
#     u,v,p = Us[t], Vs[t], Ps[t]
#     # print(u)
#     fig, axes = plt.subplots(nrows=3,figsize=(8,9))
#     fig.suptitle(f'Tims: {t*dt:0.3f} (s)')
#     fig.subplots_adjust(hspace=0.2, wspace=0.2)
#     for i,ax in enumerate(axes):
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_xlim([0.00, 2])
#         ax.set_ylim([0.00, 2])
#         ax.set_title(Titles[i])
#         for key, spine in ax.spines.items():
#             if key in ['right','top','left','bottom']:
#                 spine.set_visible(False)
#         ax.axis('equal')
#     axu = axes[0];axv = axes[1];axp = axes[2]
#     # u = u.reshape(-1)
#     # v = v.reshape(-1)
#     # p = p.reshape(-1)
#     cf = axu.contourf(X,Y,u, alpha=0.5, cmap=cm.viridis,levels=np.linspace(umin,umax,30))#, marker='o', s=2
#     divider = make_axes_locatable(axu)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     fig.colorbar(cf, cax=cax)#, fraction=0.046, pad=0.04)

#     cf = axv.contourf(X,Y,v, alpha=0.5, cmap=cm.viridis,levels=np.linspace(vmin,vmax,30))#, marker='o', s=2
#     divider = make_axes_locatable(axv)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     fig.colorbar(cf, cax=cax)

#     cf = axp.contourf(X,Y,p, alpha=0.5, cmap=cm.viridis,levels=levels)#, marker='o', s=2
#     divider = make_axes_locatable(axp)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     fig.colorbar(cf, cax=cax)
#     fig.savefig(f'Figures/Separated/Time{t:03d}.png',bbox_inches='tight')
#     plt.close('all')

#     fig, ax = plt.subplots()
#     fig.set_size_inches(11, 7)
#     div = make_axes_locatable(ax)
#     cax = div.append_axes('right', '5%', '5%')

#     cf = ax.contourf(X,Y,p,alpha= 0.5, cmap=cm.viridis,levels=levels);
#     cb = fig.colorbar(cf, cax=cax)
#     cr = ax.contour(X,Y,p, cmap=cm.viridis,vmin=-4,vmax=4);
#     skip = 2
#     cq = ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], u[::skip, ::skip], v[::skip, ::skip])
#     ax.set_xlabel('$x$')
#     ax.set_ylabel('$y$');
#     ax.text(0.3,1.05,  "Lid Cavity Flow Pressure Plot Time history", transform=ax.transAxes);
#     fig.savefig(f'Figures/Together/Time{t:03d}.png',bbox_inches='tight')
#     del u, v, p

X = X.reshape(-1)
Y = Y.reshape(-1)
Total = np.empty(shape=[0,6])
for t in range(nt):
    u,v,p = Us[t],Vs[t],Ps[t]
    u = u.reshape(-1);v = v.reshape(-1);p = p.reshape(-1)
    T = t*np.ones_like(X)
    Dest = np.vstack((T,X,Y,u,v,p)).T
    Total = np.vstack((Total,Dest))
print(Total.shape)

np.savetxt('results.csv', Total, delimiter=',') 


imgs = glob.glob('Figures/Together/*.png')
imgs.sort()

frames = []

for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

frames[0].save(f'Figures/Together/Evolution.gif', format='GIF',
                append_images=frames[1:],save_all=True,
                duration=1200, loop=0)