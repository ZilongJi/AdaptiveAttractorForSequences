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
position = position.reshape((-1, 1))
Iext = cann.get_stimulus_by_pos(position)

runner = bp.DSRunner(cann,
                     inputs=('input', Iext, 'iter'),
                     monitors=['u', 'v', 'r'])

runner.run(dur)
fig, axes = plt.subplots(nrows=2) # 创建两个子图
axes[0].plot([1, 2, 3], [4, 5, 6]) # 在第一个子图上画线
axes[0].get_xaxis().set_visible(False) # 隐藏第一个子图的x轴
axes[0].get_yaxis().set_visible(False) # 隐藏第一个子图的y轴
# 设置坐标轴的线条粗细
axes[1].spines['top'].set_linewidth(1)
axes[1].spines['right'].set_linewidth(1)
axes[1].spines['bottom'].set_linewidth(1)
axes[1].spines['left'].set_linewidth(1)
index = np.linspace(1, cann.num, cann.num)
# index = np.linspace(0, 300, 300)
fr = runner.mon.r
pos = position.squeeze()
# plt.pcolormesh(index,position, fr[100:400,:])
im = axes[1].pcolormesh(index, time[2000:12000]-time[2000], 1e3*fr[2000:12000,:], cmap='jet')
plt.xlabel('Cell indices', fontsize=15)
# 添加颜色条
clb = fig.colorbar(im, ax=axes[1])
clb.set_label('Firing rate(spikes/s)', fontsize=15)
# 设置xtick和ytick的取值
xticks = np.linspace(1,np.max(index),4)
axes[1].set_xticks(xticks)
axes[1].set_yticks([])
# fig.savefig('Figures/place_field.png', dpi=300)
plt.tight_layout()
plt.show()
