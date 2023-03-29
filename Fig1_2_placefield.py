import brainpy as bp
import brainpy.math as bm
import numpy as np
import jax
import seaborn as sns
import matplotlib.pyplot as plt
from cann_fft import CANN1D
bm.set_platform('cpu')

cann = CANN1D(num=64, mbar=1)
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
aspect_ratio = 4/3
ylen = 5
fig, ax = plt.subplots(figsize=(4, 6))
# 设置坐标轴的线条粗细
ax.spines['top'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
index = np.linspace(1, cann.num, cann.num)
# index = np.linspace(0, 300, 300)
fr = runner.mon.r.T
pos = position.squeeze()
# plt.pcolormesh(index,position, fr[100:400,:])
plt.pcolormesh(pos[2000:12000:20]-pos[2000], index[10:50]-index[10], 1e3*fr[10:50,2000:12000:20], cmap='viridis')

# plt.pcolormesh(position, index,  runner.mon.u)
plt.xlabel('Position', fontsize=18)
plt.ylabel('Cell indices', fontsize=18)
clb = plt.colorbar()
clb.set_label('Firing rate(spikes/s)', fontsize=18)
tick_size = 15
# xticks = np.linspace(0,np.max(vext),4)
yticks = [0,15,30]
# ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.tick_params(axis='x', labelsize=tick_size)
ax.tick_params(axis='y', labelsize=tick_size)
# clb.ax.set_title('Firing rate(spikes/s)')
plt.tight_layout()
plt.show()
fig.savefig('Figures/place_field.png', dpi=300)
fig.savefig('Figures/place_field.pdf', dpi=300)
