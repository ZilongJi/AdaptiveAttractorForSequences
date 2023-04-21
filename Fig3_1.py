import brainpy as bp
import brainpy.math as bm
import numpy as np
import jax
import seaborn as sns
import matplotlib.pyplot as plt
from cann import CANN1D
import scipy
bm.set_platform('cpu')

cann = CANN1D(tau=3, tau_v=144., num=128, mbar=150)
v_ext = cann.a / cann.tau_v * 0.55
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
noise = 0.01*np.random.randn(num,cann.num)
Iext = cann.get_stimulus_by_pos(position) + noise
# Iext = cann.get_stimulus_by_pos(position)

runner = bp.DSRunner(cann,
                     inputs=('input', Iext, 'iter'),
                     monitors=['u', 'v', 'r','center','centerI'])

runner.run(dur)
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
fig0, ax0 = plt.subplots(figsize=(6,6))
plt.plot(time-time[0], relative_pos)
plt.scatter(time[Peaks]-time[0], relative_pos[Peaks])

##Visualize theta sweeps
ylen = 6
label_size = 18
tick_size = 15
fig, ax = plt.subplots(figsize=(8,4))
# 设置所有线条粗细
ax.spines['top'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
im = plt.pcolormesh(time[0:-1:50]-time[0], pos, fr[:,0:-1:50]*1e3, cmap='viridis')
clb = plt.colorbar(im)
clb.set_label('Firing rate(spikes/s)', fontsize=label_size)
plt.plot(time-time[0], cI, color='r', linewidth=2)
# plt.plot(time-time[0], cU, color='w', linewidth=2)
for peaks in Peaks:
    plt.plot([time[peaks]-time[0], time[peaks]-time[0]],[-np.pi,np.pi],'w--', linewidth=1)
plt.xlim(0, 1e3)
# plt.xlabel('time(ms)', fontsize=label_size)
plt.ylabel('Decoded Position', fontsize=label_size)
# 设置xtick和ytick的取值
xticks = np.linspace(0, 1e3, 3)
yticks = np.linspace(-2.5,2.5,3)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
# 设置xtick和ytick的字体大小
ax.tick_params(axis='x', labelsize=tick_size)
ax.tick_params(axis='y', labelsize=tick_size)
plt.ylim([-2.5, 2.5])
plt.tight_layout()
plt.show()
fig.savefig('Figures/Fig3_1.png', dpi=300)
fig.savefig('Figures/Fig3_1.pdf', dpi=300)


fig, ax = plt.subplots(figsize=(8,4))
# 设置所有线条粗细
ax.spines['top'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
t_start = Trough[1]
t_end = Trough[2]

im = plt.pcolormesh(time[t_start:(t_end)]-time[t_start], pos-cI[t_start], fr[:,t_start:(t_end)]*1e3, cmap='viridis')
clb = plt.colorbar(im)
clb.set_label('Firing rate(spikes/s)', fontsize=label_size)
# plt.plot(time-time[0], cI, color='r', linewidth=2)
plt.plot([time[Peaks[1]]-time[t_start], time[Peaks[1]]-time[t_start]],[-np.pi,np.pi],'w--', linewidth=3)
plt.plot([time[Peaks[2]]-time[t_start], time[Peaks[2]]-time[t_start]],[-np.pi,np.pi],'w--', linewidth=3)
plt.plot([time[Peaks[3]]-time[t_start], time[Peaks[3]]-time[t_start]],[-np.pi,np.pi],'w--', linewidth=3)

# plt.xlabel('time(ms)', fontsize=label_size)
plt.ylabel('Decoded Position', fontsize=label_size)
# 设置xtick和ytick的取值
xticks = np.linspace(0, 88, 3)
yticks = np.array([-1.2,0,1.5])
ax.set_xticks(xticks)
ax.set_yticks(yticks)
# 设置xtick和ytick的字体大小
ax.tick_params(axis='x', labelsize=tick_size)
ax.tick_params(axis='y', labelsize=tick_size)
plt.xlim(0, 88)
plt.ylim([-1.2,1.5])
plt.tight_layout()
plt.show()
fig.savefig('Figures/Fig3_1a.png', dpi=300)
fig.savefig('Figures/Fig3_1a.pdf', dpi=300)




# encoder = bp.encoding.PoissonEncoder()
# spike = encoder(fr.T*1e2)
# plt.figure()
# bp.visualize.raster_plot(time[probe_num:-1],spike)
# plt.show()