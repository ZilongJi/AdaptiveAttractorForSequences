import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from cann import CANN1D
import scipy
#set default ramndom seed for reproducibility
np.random.seed(0)
#set backend to cpu
bm.set_platform('cpu')

#build and run the network
# cann = CANN1D(tau=3, tau_v=144., num=128, mbar=150, A=0.01)
cann = CANN1D(tau=1., tau_v=48., num=128, mbar=7.5, A=0.01)
v_ext = cann.a / 144 * 0.55
dur = 2.5*np.pi / v_ext

dt = bm.get_dt()
num = int(dur / dt)
time = np.linspace(0, dur, num)
final_pos = v_ext * dur
position = np.zeros(num)
position[0] = -np.pi/3
for i in range(num)[1:]:
    position[i] = position[i - 1] + v_ext * dt
    if position[i] > np.pi:
        position[i] -= 2 * np.pi

position = position.reshape((-1, 1))
noise = 0.0*np.random.randn(num,cann.num)
Iext = cann.get_stimulus_by_pos(position) + noise

runner = bp.DSRunner(cann,
                     inputs=('input', Iext, 'iter'),
                     monitors=['u', 'v', 'r','center','centerI'])

runner.run(dur)

# fig, ax = plt.subplots(figsize=(4,2))
# ax.plot(runner.mon.center)
# ax.plot(runner.mon.centerI)
# plt.show()



probe_num = int( 1.9*bm.pi / v_ext/dt)
time=time[probe_num:-1]
index = np.linspace(1, cann.num, cann.num)
pos = np.linspace(-np.pi,np.pi,cann.num+1)
pos = pos[0:-1]
fr = runner.mon.r.T[:,probe_num:-1]
cI = runner.mon.centerI[probe_num:-1]
cU = runner.mon.center[probe_num:-1]
##Theta oscillations
relative_pos = runner.mon.center[probe_num:-1] - runner.mon.centerI[probe_num:-1]
relative_pos = bm.where(relative_pos > np.pi, relative_pos - 2*np.pi,  relative_pos)
relative_pos = bm.where(relative_pos < -np.pi, relative_pos + 2*np.pi,  relative_pos)
relative_pos = np.squeeze(relative_pos)
Peaks,_ = scipy.signal.find_peaks(relative_pos, width=300)
Trough,_ = scipy.signal.find_peaks(-relative_pos, width=300)

#visualize bump sweeps
fig, ax = plt.subplots(figsize=(4,2))
#set parameters
labelsize = 10
ticksize = 8
position_color = 'darkturquoise'

plt.pcolormesh(time[0:-1:50]-time[0], pos, fr[:,0:-1:50]*1e3, cmap='inferno')
clb = plt.colorbar(ticklocation='right', ticks=[0,1,2])
clb.set_label('Firing rate (Hz)', fontsize=labelsize)
clb.ax.tick_params(labelsize=ticksize)
#add animal position to the plot 
plt.plot(time-time[0], cI, color=position_color, linewidth=1)  

#add separate lines to theta sweeps
for peaks in Peaks:
    plt.plot([time[peaks]-time[0], time[peaks]-time[0]],[-np.pi,np.pi],'w--', linewidth=1)

plt.xlim(0, 1e3)
plt.ylim([-2.5, 2.5])
plt.xlabel('Time (s)', fontsize=labelsize)
plt.ylabel('Position (cm)', fontsize=labelsize)
# set x and y ticks
xticks = np.linspace(0, 1e3, 3)
yticks = np.linspace(-2.5,2.5,3)
ax.set_xticks(xticks)
ax.set_yticks(yticks)

#set tick labels
xticklabels = [0, 0.5, 1]
yticklabels = [0, int(2.5*100), int(5*100)]
ax.set_yticklabels(yticklabels)
ax.set_xticklabels(xticklabels)
ax.tick_params(axis='x', labelsize=ticksize)
ax.tick_params(axis='y', labelsize=ticksize)
plt.tight_layout()
fig.savefig('Figures/Fig3_thetasweep.pdf', dpi=300)

#plot the average bump sweeps
fig, ax = plt.subplots(figsize=(2,2), dpi=300)

t_start = Trough[1]
t_end = Trough[2]

plt.pcolormesh(time[t_start:(t_end)]-time[t_start], pos-cI[t_start], fr[:,t_start:(t_end)]*1e3, cmap='inferno')

'''
clb = plt.colorbar(ticklocation='right', ticks=[0,1,2])
clb.set_label('Firing rate (Hz)', fontsize=labelsize)
clb.ax.tick_params(labelsize=ticksize)
'''
#add the separate line
#plt.plot([time[Peaks[1]]-time[t_start], time[Peaks[1]]-time[t_start]],[-np.pi,np.pi],'w--', linewidth=3)
plt.plot([time[Peaks[2]]-time[t_start], time[Peaks[2]]-time[t_start]],[-np.pi,np.pi],'w--', linewidth=3)
#plt.plot([time[Peaks[3]]-time[t_start], time[Peaks[3]]-time[t_start]],[-np.pi,np.pi],'w--', linewidth=3)

#add reference line of y=0
plt.plot([0, time[t_end]-time[t_start]], [0,0], color=position_color, linewidth=1, linestyle='--')

plt.xlabel('Time (ms)', fontsize=labelsize)
plt.ylabel('Relative pos. (cm)', fontsize=labelsize)
# set the xticks and yticks
xticks = np.linspace(0, 80, 3)
yticks = np.array([-1.2,0,1.5])
ax.set_xticks(xticks)
ax.set_yticks(yticks)
# set yticklabels
yticklabels = [-int(1.2*100), 0, int(1.5*100)]
ax.set_yticklabels(yticklabels)
ax.tick_params(axis='x', labelsize=ticksize)
ax.tick_params(axis='y', labelsize=ticksize)

plt.xlim(0, 82)
plt.ylim([-1.2,1.5])
plt.tight_layout()
fig.savefig('Figures/Fig3_thetasweep_merged.pdf')