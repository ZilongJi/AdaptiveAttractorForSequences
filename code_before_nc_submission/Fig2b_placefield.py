'''
Plot the place field of cells in the CANN model.
Created on 2021-04-20 by Tianhao Chu,
MOdified by Zilong Ji
'''
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
position[0] = -np.pi
for i in range(num)[1:]:
    position[i] = position[i - 1] + v_ext * dt
    if position[i] > np.pi:
        position[i] -= 2 * np.pi

position = position.reshape((-1, 1))
noise = 0.01*np.random.randn(num,cann.num)
Iext = cann.get_stimulus_by_pos(position+0.5*np.random.randn(num,1)) + noise

runner = bp.DSRunner(cann,
                     inputs=('input', Iext, 'iter'),
                     monitors=['u', 'v', 'r'])

runner.run(dur)

fr = runner.mon.r.T
pos = position.squeeze()

# Plot the place field
# set some parameters
labelsize = 10
ticksize = 8
fig, ax = plt.subplots(figsize=(2.5, 2), dpi=300)

index = np.linspace(1, cann.num, cann.num)
#plt.pcolormesh(100*(pos[2000:12000:20]-pos[2000]), index[10:50]-index[10], 1e3*fr[10:50,2000:12000:20], cmap='inferno')
#do pcolormesh without slice
plt.pcolormesh(pos[::100]-pos[0], index-index[0], 1e3*fr[:,::100], cmap='inferno')

#add labels
plt.xlabel('Position (m)', fontsize=labelsize)
plt.ylabel('Cell index', fontsize=labelsize)

#set colorbar, add ticks 
clb = plt.colorbar(ticklocation='right', ticks=[0,1,2])
clb.set_label('Firing rate (Hz)', fontsize=labelsize)
#change the font size
clb.ax.tick_params(labelsize=ticksize)

yticks = [0,32,64,96,128]
ax.set_yticks(yticks)
#set x ticks
xticks = [0,2,4,6]
ax.set_xticks(xticks)

ax.tick_params(axis='x', labelsize=ticksize)
ax.tick_params(axis='y', labelsize=ticksize)
plt.tight_layout()

fig.savefig('Figures/Fig2b_placefield.pdf')
