#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import TwoD_gamma
import brainpy.math as bm
import scipy

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
cx = center_trace[w_start:(w_start+w_size),0]
cy = center_trace[w_start:(w_start+w_size),1]
st = 100*step[w_start:(w_start+w_size)] #change to centimeter
mr = mean_fr[w_start:(w_start+w_size)]

Peaks,_ = scipy.signal.find_peaks(-mr)

#%% plor the figure
fig1, axs = plt.subplots(2, 1, figsize=(3.5, 3), sharex=True,  dpi=300)

linecolor = '#009FB9'
# 在第一个子图中绘制折线图
axs[0].plot(st, color=linecolor)
axs[0].plot(st, color='#F18D00', marker='o', linestyle='None', markersize=2)

#add vertical lines
for peaks in Peaks:
    axs[0].plot([peaks, peaks],[0,np.max(st)],'k--', linewidth=1)

#set right and top spines invisible
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)    
    
#add ylabel
axs[0].set_ylabel('Mov. (cm)', fontsize=labelsize)
    
axs[1].plot(mr, linecolor)
for peaks in Peaks:
    axs[1].plot([peaks, peaks],[0,np.max(mr)],'k--', linewidth=1)

#set top and right spines invisible
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)

#add xlabel and ylabel and chang label to two lines
axs[1].set_xlabel('Time (ms)', fontsize=labelsize)
axs[1].set_ylabel('Slow \n Gamma', fontsize=labelsize)

#set y ticks off for the second subplot
axs[1].set_yticks([])

#align the ylabels
fig1.align_ylabels(axs)
#save the figure
fig1.savefig('Figures/fig6b.pdf', bbox_inches='tight')

# %%
