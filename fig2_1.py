import brainpy as bp
import brainpy.math as bm
import numpy as np
import jax
import seaborn as sns
import matplotlib.pyplot as plt
from cann_fft import CANN1D
from mpl_toolkits.axes_grid1 import make_axes_locatable
bm.set_platform('cpu')

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

fig, axes = plt.subplots(nrows=2,figsize=(6,4)) # 创建两个子图
axes[0].plot(index, 1e3*fr[2000,:], linewidth=2) # 在第一个子图上画线
for spine in axes[0].spines.values():
    spine.set_visible(False)
axes[0].get_xaxis().set_visible(False) # 隐藏第一个子图的x轴
axes[0].get_yaxis().set_visible(False) # 隐藏第一个子图的y轴
# 设置坐标轴的线条粗细
axes[1].spines['top'].set_linewidth(1)
axes[1].spines['right'].set_linewidth(1)
axes[1].spines['bottom'].set_linewidth(1)
axes[1].spines['left'].set_linewidth(1)
label_size = 18
tick_size = 15
# plt.pcolormesh(index,position, fr[100:400,:])
im = axes[1].pcolormesh(index, time[2000:12000:50]-time[2000], 1e3*fr[2000:12000:50,:], cmap='viridis')
plt.xlabel('Cell index', fontsize=label_size)

xticks = np.array([1,32,64,96,128])
axes[1].set_xticks(xticks)
axes[1].tick_params(axis='x', labelsize=tick_size)
axes[1].set_yticks([])
plt.tight_layout()
plt.show()
fig.savefig('Figures/Fig2_1.png', dpi=300)
fig.savefig('Figures/Fig2_1.pdf', dpi=300)

# plt.show()
