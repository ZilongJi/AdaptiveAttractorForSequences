#%%
import numpy as np
import matplotlib.pyplot as plt
import TwoD_gamma
import brainpy.math as bm
#set default ramndom seed for reproducibility
bm.random.seed(1)
#set backend to cpu
bm.set_platform('cpu')

labelsize = 18
ticksize = 14

center_trace, step, mean_fr = TwoD_gamma.get_trace(duration=3e4, sample_rate=20, T_start=1000, visual=False)
w_start = 95
w_size = 160
w_step = 3
cx = 100*center_trace[w_start:(w_start+w_size),0]
cy = 100*center_trace[w_start:(w_start+w_size),1]
st = 100*step[w_start:(w_start+w_size)]
mr = mean_fr[w_start:(w_start+w_size)]

#%%
#substract the minimum in cx and cy
cx = cx - np.min(cx)
cy = cy - np.min(cy)

linecolor = '#009FB9'
#plot figure 1
x = cx[0:-1:w_step]
y = cy[0:-1:w_step]
fig1, ax = plt.subplots(figsize=(4, 3), dpi=300)
size = 128
sigma = 20
xx = np.linspace(np.min(x)-10,np.max(cx)+10,size)
yy = np.linspace(np.min(y)-10,np.max(cy)+10,size)
X,Y = np.meshgrid(xx,yy)
Z = np.zeros((size,size))
for i in range(x.shape[0]):
    Z += np.exp((-(X-x[i])**2-(Y-y[i])**2)/(2*sigma**2))
#plot x and y and set the line color to linecolor
ax.plot(x, y, color=linecolor, linewidth=2)
ax.contourf(X, Y, Z, alpha=1, levels=100,cmap='inferno')
#set limits

ax.set_xlim(np.min(x)-10,np.max(cx)+10)
ax.set_ylim(np.min(y)-10,np.max(cy)+10)
#add ticks
ax.set_xticks([0, 100, 200, 300])
ax.set_yticks([0, 100, 200, 300, 400])
ax.set_xlabel('X position (cm)', fontsize=labelsize)
ax.set_ylabel('Y position (cm)', fontsize=labelsize)
#set tick size
ax.tick_params(labelsize=ticksize)
plt.tight_layout()
#save figure
plt.savefig('./Figures/Fig6a_1.pdf', bbox_inches='tight')

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
        plt.savefig('Figures/Fig6a/Fig6a_'+str(i)+'.pdf', bbox_inches='tight')

fig2, axs = plt.subplots(3, 1, figsize=(4, 3), dpi=300)
#   
axs[0].plot(cx, color=linecolor)
#make the x axis invisible
axs[0].axes.xaxis.set_visible(False)
#add y label
axs[0].set_ylabel('X Pos.', fontsize=labelsize)
#remove the TOP AND RIGHT axis 
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

axs[1].plot(cy, color=linecolor)
#add y label
axs[1].set_ylabel('Y Pos.', fontsize=labelsize)
#make the x axis invisible
axs[1].axes.xaxis.set_visible(False)
#remove the top and right axis 
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

axs[2].plot(st, color=linecolor)
axs[2].plot(st, color='#F18D00', marker='o', linestyle='None', markersize=2)
#add y label and algin the label to the left
axs[2].set_ylabel('Mov.', fontsize=labelsize)
#add x labels
axs[2].set_xlabel('Time (ms)', fontsize=labelsize)
#remove top right axis
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)

#set tick size
axs[0].tick_params(axis='both', which='major', labelsize=ticksize)
axs[1].tick_params(axis='both', which='major', labelsize=ticksize)
axs[2].tick_params(axis='both', which='major', labelsize=ticksize)

#align the y labels of three subplots   
fig2.align_ylabels(axs[:])  
#plt.tight_layout()
#save figure
plt.savefig('./Figures/Fig6a_2.pdf', bbox_inches='tight')


# %%
