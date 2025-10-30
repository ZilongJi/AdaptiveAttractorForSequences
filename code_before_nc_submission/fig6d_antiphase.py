
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

center_trace, step, mean_fr, spike_num = TwoD_gamma.get_trace(duration=3e4, beta=0.2, sample_rate=20, T_start=2000, T_sample=1000, visual=False, m_0 = 0.7)
w_start = 95
w_size = 160
w_step = 3
period = 20
phase_step = np.zeros(period)
phase_r = np.zeros(period)
phase = np.linspace(-np.pi, np.pi, period + 1)[:-1]
for i in range(int(len(mean_fr) / period)):
    for j in range(period):
        phase_step[j] = phase_step[j] + step[i * period + j]
        phase_r[j] = phase_r[j] + mean_fr[i * period + j]
phase_r = (phase_r-np.min(phase_r))/(np.max(phase_r)-np.min(phase_r))
phase_step = (phase_step-np.min(phase_step))/(np.max(phase_step)-np.min(phase_step))
phase = np.append(phase, phase[0])
phase_r = np.append(phase_r, phase_r[0])
phase_step = np.append(phase_step, phase_step[0])
fig = plt.figure(figsize=(3,3), dpi=300)
ax = fig.add_subplot(111, projection='polar')
ax.plot(phase, phase_r, color='#009FB9')
ax.plot(phase, phase_step, color='#F18D00')
#add grid
ax.grid(True)

#set tick size
ax.tick_params(labelsize=ticksize)
ax.set_rticks([0, 0.2, 0.4, 0.6, 0.8])
#reduce rticks size
for label in ax.get_yticklabels():
    label.set_fontsize(ticksize/1.5)
ax.set_rlabel_position(60)
#save figure
fig.savefig('./Figures/Fig6d.pdf', bbox_inches='tight')

threshold = 0.04
indices = np.where(step < threshold)
index = np.where(step >= threshold)
spike_low = np.array(spike_num[indices])
spike_high = np.array(spike_num[index])
bins = np.linspace(0, 12, 11)
hist_low, edges_l = np.histogram(spike_low, bins=bins)
hist_high, edges_h = np.histogram(spike_high, bins=bins)
bin_centers_l = 0.5 * (edges_l[:-1] + edges_l[1:])
bin_centers_h = 0.5 * (edges_h[:-1] + edges_h[1:])
hist_low = hist_low / np.sum(hist_low)
hist_high = hist_high / np.sum(hist_high)
fig2, axes = plt.subplots(figsize=(6, 4))
# bins = [-3, -2, -1, 0, 1, 2, 3]
axes.plot(bin_centers_l,hist_low)
axes.plot(bin_centers_h,hist_high)

plt.show()
