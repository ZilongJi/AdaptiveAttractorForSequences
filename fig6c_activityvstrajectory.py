#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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
cx = center_trace[w_start:(w_start + w_size), 0]
cy = center_trace[w_start:(w_start + w_size), 1]
st = step[w_start:(w_start + w_size)]
mr = mean_fr[w_start:(w_start + w_size)]
period = int(20)
phase_step = np.zeros(period)
phase_r = np.zeros(period)
phase = np.linspace(-np.pi, np.pi, period+1)[:-1]
for i in range(int(len(mean_fr) / period)):
    for j in range(period):
        phase_step[j] = phase_step[j] + step[i * period + j]
        phase_r[j] = phase_r[j] + mean_fr[i * period + j]

#%%
fig1, axs = plt.subplots(2, 1, figsize=(3.5, 3), sharex=True, dpi=300)

#add another cycle to the end of the trace
phase1 = phase + np.pi
phase2 = phase+3*np.pi
phase_r1 = phase_r
phase_r2 = phase_r
phase_step1 = phase_step
phase_step2 = phase_step

#concate the two cycles
phase12 = np.append(phase1, phase2)
phase_r12 = np.append(phase_r2, phase_r2)
phase_step12 = np.append(phase_step1, phase_step2)

axs[0].plot(phase12, phase_r12, color='k')
down_sample = 1
axs[0].bar(phase12[0:-1:down_sample], phase_r12[0:-1:down_sample], width=1.2*np.pi/period, color='k', alpha=0.3)  
#add y label
axs[0].set_ylabel('Activity', fontsize=labelsize)
#set y ticks to scintific notation
axs[0].set_yticks([0, 0.05])
axs[0].set_yticklabels(['0', '0.05'], fontsize=ticksize)

axs[1].plot(phase12[0:-1], phase_step12[0:-1], color='k')
axs[1].bar(phase12[0:-1:down_sample], phase_step12[0:-1:down_sample], width=1.2*np.pi/period, color='k', alpha=0.3)
#add y label
axs[1].set_ylabel('Step size', fontsize=labelsize)

#add x label
axs[1].set_xlabel('Slow gamma phase', fontsize=labelsize)

#add xticks
axs[1].set_xticks([0, 2*np.pi, 4*np.pi])
#add xticklabels and yticklabels
axs[1].set_xticklabels(['0', '$360^o$', '$720^o$'], fontsize=ticksize)

#set y tick and fontszie
axs[1].set_yticks([0, 5, 10])
axs[1].set_yticklabels(['0', '5', '10'], fontsize=ticksize)
#align y labels
fig1.align_ylabels(axs)
plt.tight_layout()
#save figure
fig1.savefig('Figures/Fig6c.pdf', bbox_inches='tight')

