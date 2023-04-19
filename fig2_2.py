import brainpy as bp
import brainpy.math as bm
import numpy as np
import jax
import seaborn as sns
import matplotlib.pyplot as plt
from cann_fft import CANN1D
bm.set_platform('cpu')

cann = CANN1D(num=256, mbar=0, tau=3, tau_v=144)
v_ext = cann.a / cann.tau_v * 10
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
                     monitors=['u', 'v', 'r','center'])

runner.run(dur)
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
label_size = 18
tick_size = 15
# 设置所有线条粗细
ax.spines['top'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
index = np.linspace(1, cann.num, cann.num)
# index = np.linspace(0, 300, 300)
fr = runner.mon.r.T
cU = runner.mon.center
pos = np.linspace(-np.pi,np.pi,cann.num)
# plt.pcolormesh(index,position, fr[100:400,:])
tstart = 300
tend = -300
im = plt.pcolormesh(time[tstart:tend:25]-time[tstart], pos[50:180] - pos[50], 1e3*fr[50:180,tstart:tend:25], cmap='viridis')
time_slice = 500
plt.scatter(time[time_slice:-time_slice-50:100]-time[tstart], position[time_slice:-time_slice-50:100]-position[time_slice]+0.1, marker='v',color = 'k')
plt.scatter(time[time_slice:-time_slice-50:100]-time[tstart], cU[time_slice:-time_slice-50:100]-position[time_slice], marker='^', color='white', edgecolor='none')
plt.plot(time[time_slice:-time_slice-50]-time[tstart], position[time_slice:-time_slice-50]-position[time_slice], 'b', linewidth=2)
# 设置xtick和ytick的取值
xticks = np.linspace(0,150,4)
yticks = np.linspace(0,3,4)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
plt.ylim([0,3.14])
plt.xlim([0,150])
# 设置xtick和ytick的字体大小
ax.tick_params(axis='x', labelsize=tick_size)
ax.tick_params(axis='y', labelsize=tick_size)
plt.xlabel('time (ms)', fontsize=label_size)
plt.ylabel('Decoded position (rads)', fontsize=label_size)
clb = plt.colorbar(im)
clb.set_label('Firing rate (spikes/s)', fontsize=label_size)
cticks = np.linspace(0,2,5)
clb.set_ticks(cticks)
# clb.ax.set_title('Firing rate(spikes/s)')
plt.tight_layout()
plt.show()
fig.savefig('Figures/Fig2_2.png', dpi=300)
fig.savefig('Figures/Fig2_2.pdf', dpi=300)
