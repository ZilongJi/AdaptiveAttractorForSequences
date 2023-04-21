import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from cann import CANN1D
bm.set_platform('cpu')

#create and run the network
cann = CANN1D(num=256, mbar=0, tau=3, tau_v=144)
v_ext = cann.a / cann.tau_v * 10
dur = 2 * bm.pi / v_ext
dt = bm.get_dt()
num = int(dur / dt)
time = np.linspace(0, dur, num)*10
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
                     monitors=['u', 'v', 'r','center'])

runner.run(dur)


# Plot the figure
fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(1, 1, 1)
#set some parameters
labelsize = 18
ticksize = 14
position_color = '#F18D00'
bump_color = '#009FB9'

index = np.linspace(1, cann.num, cann.num)

fr = runner.mon.r.T
cU = runner.mon.center
pos = np.linspace(-np.pi,np.pi,cann.num)

tstart = 300
tend = -300
time /= 1000
im = plt.pcolormesh(time[tstart:tend:25]-time[tstart], (pos[50:180] - pos[50])*1e2, 1e3*fr[50:180,tstart:tend:25], cmap='inferno')
time_slice = 500
plt.scatter(time[time_slice:-time_slice-50:100]-time[tstart], (position[time_slice:-time_slice-50:100]-position[time_slice])*1e2, marker='o',color = position_color, edgecolor='k')
plt.scatter(time[time_slice:-time_slice-50:100]-time[tstart], (cU[time_slice:-time_slice-50:100]-position[time_slice])*1e2, marker='o', color=bump_color, edgecolor='k')
plt.plot(time[time_slice:-time_slice-50]-time[tstart], (position[time_slice:-time_slice-50]-position[time_slice])*1e2, color ='k', linewidth=1, linestyle='--')

#set x and y ticks
xticks = np.linspace(0,1500/1000,4)
yticks = np.linspace(0,300,4)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
plt.ylim([0,314])
plt.xlim([0,1500/1000])
# set font size on x and y ticks
ax.tick_params(axis='x', labelsize=ticksize)
ax.tick_params(axis='y', labelsize=ticksize)
plt.xlabel('time (s)', fontsize=labelsize)
plt.ylabel('Position (cm)', fontsize=labelsize)
clb = plt.colorbar(im, ticks=[0,1,2])
clb.set_label('Firing rate (Hz)', fontsize=labelsize)
clb.ax.tick_params(labelsize=ticksize)
plt.tight_layout()

fig.savefig('Figures/Fig2_2.pdf', dpi=300)
