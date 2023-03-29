import brainpy as bp
import brainpy.math as bm
import numpy as np
import jax
import seaborn as sns
import matplotlib.pyplot as plt
from cann import CANN1D
import scipy
bm.set_platform('cpu')

cann = CANN1D(tau=3, tau_v=144., num=128, mbar=150.3)
v_ext = cann.a / cann.tau_v * 0.55
dur =  2.5*np.pi / v_ext
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
probe_num = int( 1.6*bm.pi / v_ext/dt)
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
Peaks_neg,_ = scipy.signal.find_peaks(-relative_pos, width=300)
speed = np.diff(relative_pos)
fig0, ax0 = plt.subplots(figsize=(6,6))
plt.plot(time-time[0], relative_pos)
plt.scatter(time[Peaks]-time[0], relative_pos[Peaks], color='blue')
plt.scatter(time[Peaks_neg]-time[0], relative_pos[Peaks_neg], color='red')
print(len(Peaks))
print(len(Peaks_neg))


fr_pos = np.zeros(len(Peaks)-1)
fr_neg = np.zeros(len(Peaks)-1)
for i in range(len(Peaks)-1):
    fr_neg[i] = np.mean(fr[:,Peaks[i]:Peaks_neg[i+1]])
    fr_pos[i] = np.mean(fr[:,Peaks_neg[i]:Peaks[i]])
std_pos = np.std(fr_pos)
std_neg = np.std(fr_neg)
label_size = 18
tick_size = 15
fig, ax = plt.subplots(figsize=(6,4))
# 设置所有线条粗细
ax.spines['top'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
# x变量
x = ['forward window', 'Reverse window']
# y变量
y = np.array([np.mean(fr_pos)*1e3, np.mean(fr_neg)*1e3])
std_fr = np.array([std_pos*1e3, std_neg*1e3])
ax.errorbar(x, y, yerr = std_fr, fmt='o', color='blue', ecolor='black', capsize=5, capthick=2)
# 画直方图
plt.bar(x, y)
plt.ylabel('Mean firing rates (spikes/s)', fontsize=label_size)
# 设置xtick和ytick的字体大小
ax.tick_params(axis='x', labelsize=tick_size)
ax.tick_params(axis='y', labelsize=tick_size)
# plt.ylim([-2.5, 2.5])
plt.tight_layout()
plt.show()
fig.savefig('Figures/Fig3_3_2.png', dpi=300)

