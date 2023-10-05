import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from cann_fft import CANN1D
#set default ramndom seed for reproducibility
np.random.seed(0)
#set backend to cpu
bm.set_platform('cpu')

#create and run the network
cann = CANN1D(num=128, mbar=1)
v_ext = cann.a / cann.tau_v * 0.5
dur = 2 * bm.pi / v_ext
dt = bm.get_dt()
num = int(dur / dt)
time = np.linspace(0, dur, num)
final_pos = v_ext * dur
position = np.zeros(num)
position = position.reshape((-1, 1))
Iext = cann.get_stimulus_by_pos(position)

runner = bp.DSRunner(cann,
                     inputs=('input', Iext, 'iter'),
                     monitors=['u', 'v', 'r'])

runner.run(dur)
index = np.linspace(1, cann.num, cann.num)
# index = np.linspace(0, 300, 300)
fr = runner.mon.r
pos = position.squeeze()

#plot the figrue
# set some parameters
labelsize = 10
ticksize = 8
custom_color = '#009FB9'

#create twwo subplots with the first one is bigger than the second one
fig, axes = plt.subplots(nrows=2, figsize=(2.5,2), height_ratios=[2, 1], dpi=300)
#plot the line and set the line color to customized color
axes[0].plot(index, 1e3*fr[2000,:], linewidth=2, color=custom_color) 
#set axis off and box off in this subplot
axes[0].set_xticks([])
axes[0].set_yticks([])
for spine in axes[0].spines.values():
    spine.set_visible(False)
    
# plt.pcolormesh(index,position, fr[100:400,:])
im = axes[1].pcolormesh(index, time[2000:12000:50]-time[2000], 1e3*fr[2000:12000:50,:], cmap='inferno')
plt.xlabel('Cell index', fontsize=labelsize)

xticks = np.array([1,32,64,96,128])
axes[1].set_xticks(xticks)
axes[1].tick_params(axis='x', labelsize=ticksize)
axes[1].set_yticks([])
plt.tight_layout()

fig.savefig('Figures/Fig2a_networkbump.pdf', bbox_inches='tight')