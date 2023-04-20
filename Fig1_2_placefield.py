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

fig, ax = plt.subplots(figsize=(4, 6), dpi=300)

index = np.linspace(1, cann.num, cann.num)

fr = runner.mon.r.T
pos = position.squeeze()

plt.pcolormesh(pos[2000:12000:20]-pos[2000], index[10:50]-index[10], 1e3*fr[10:50,2000:12000:20], cmap='viridis')
plt.xlabel('Position (m)', fontsize=12)
plt.ylabel('Cell index', fontsize=12)
clb = plt.colorbar()
clb.set_label('Firing rate (Hz)', fontsize=12)
# xticks = np.linspace(0,np.max(vext),4)
yticks = [0,20,40]
# ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
plt.tight_layout()
#plt.show()
fig.savefig('Figures/place_field.pdf')
