import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import TwoD_gamma
import brainpy.math as bm
#set default ramndom seed for reproducibility
np.random.seed(0)
#set backend to cpu
bm.set_platform('cpu')

labelsize = 18
ticksize = 14

center_trace, step, mean_fr = TwoD_gamma.get_trace(duration=3e4, sample_rate=20, T_start=1000, visual=False)
w_start = 95
w_size = 160
w_step = 3
cx = center_trace[w_start:(w_start+w_size),0]
cy = center_trace[w_start:(w_start+w_size),1]
st = step[w_start:(w_start+w_size)]
mr = mean_fr[w_start:(w_start+w_size)]

#%%plot figures
fig1, axs = plt.subplots(3, 1, figsize=(8, 3))
#   
axs[0].plot(100*cx)
#make the x axis invisible
axs[0].axes.xaxis.set_visible(False)
axs[1].plot(100*cy)
#add y label
axs[1].set_ylabel('Y Pos. (cm)', fontsize=labelsize)
#make the x axis invisible
axs[1].axes.xaxis.set_visible(False)
axs[2].plot(100*st)

plt.tight_layout()

x = cx[0:-1:w_step]
y = cy[0:-1:w_step]
fig2, ax = plt.subplots(figsize=(3, 3), dpi=300)
size = 128
sigma = 0.05
xx = np.linspace(np.min(x)-0.1,np.max(cx)+0.1,size)
yy = np.linspace(np.min(y)-0.1,np.max(cy)+0.1,size)
X,Y = np.meshgrid(xx,yy)
Z = np.zeros((size,size))
for i in range(x.shape[0]):
    Z += np.exp((-(X-x[i])**2-(Y-y[i])**2)/(2*sigma**2))
ax.plot(x, y, linewidth=2)
ax.contourf(X, Y, Z, alpha=1, levels=100,cmap='inferno')
#set limits
ax.set_xlim(np.min(x)-0.1,np.max(cx)+0.1)
ax.set_ylim(np.min(y)-0.1,np.max(cy)+0.1)
#add ticks
ax.set_xticks([-1.0,0])
ax.set_xticklabels([0,100], fontsize=ticksize)
ax.set_yticks([-1.0,2])
ax.set_yticklabels([0,300], fontsize=ticksize)
#ad xlabel and ylabel   
ax.set_xlabel('X position (cm)', fontsize=labelsize)
ax.set_ylabel('Y position (cm)',    fontsize=labelsize)

#save figure
plt.savefig('Fig5a.png', bbox_inches='tight')

#%plot each of the sub figures
#create a folder to save figures
import os
if not os.path.exists('Figures/Fig6a'):
    os.makedirs('Figures/Fig6a')
    
Z = np.zeros((size,size))
for i in range(x.shape[0]):
    if i%8 == 0:
        Z = np.exp((-(X-x[i])**2-(Y-y[i])**2)/(2*sigma**2))
        fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
        ax.contourf(X, Y, Z, alpha=1, levels=100,cmap='inferno')
        #set axis off
        ax.axis('off')
        #save figure to the folder
        plt.savefig('Figures/Fig6a/Fig6a_'+str(i)+'.png', bbox_inches='tight')
        
# %%
