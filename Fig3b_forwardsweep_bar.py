import brainpy as bp
import brainpy.math as bm
import numpy as np
import jax
import seaborn as sns
import matplotlib.pyplot as plt
from cann import CANN1D
import scipy
from scipy.stats import ttest_ind
#set default ramndom seed for reproducibility
np.random.seed(0)
#set backend to cpu
bm.set_platform('cpu')

cann = CANN1D(tau=3, tau_v=144., num=128, mbar=153)
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
# probe_num = int( 1.9*bm.pi / v_ext/dt)#if mbar=153
probe_num = int( 1.6*bm.pi / v_ext/dt)#if mbar=150.3
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

'''
fig0, ax0 = plt.subplots(figsize=(6,6))
plt.plot(time-time[0], relative_pos)
plt.scatter(time[Peaks]-time[0], relative_pos[Peaks], color='blue')
plt.scatter(time[Peaks_neg]-time[0], relative_pos[Peaks_neg], color='red')
print(len(Peaks))
print(len(Peaks_neg))
'''

fr_pos = np.zeros(len(Peaks)-1)
fr_neg = np.zeros(len(Peaks)-1)
for i in range(len(Peaks)-1):
    fr_neg[i] = np.mean(fr[:,Peaks[i]:Peaks_neg[i+1]])
    fr_pos[i] = np.mean(fr[:,Peaks_neg[i]:Peaks[i]])
# 对两组数据进行双样本 t-检验
t_statistic, p_value = ttest_ind(fr_pos, fr_neg)

# 输出结果
print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")

std_pos = np.std(fr_pos)
std_neg = np.std(fr_neg)

fig, ax = plt.subplots(figsize=(3,3), dpi=300)
#set parameters for seaborn
labelsize = 18
ticksize = 16
bar_color = '#009FB9'

# x变量
x = ['Forward', 'Reverse']
# y变量
y = np.array([np.mean(fr_pos)*1e3, np.mean(fr_neg)*1e3])
std_fr = np.array([std_pos*1e3, std_neg*1e3])
#plot errorbar
ax.errorbar(x, y, yerr = std_fr, fmt='o', color='k', ecolor='black', capsize=5, capthick=1)

#plot bar with bar_color
ax.bar(x, y, color=bar_color, width=0.5, alpha=1)
plt.ylabel('Bump height', fontsize=labelsize)
# 设置xtick和ytick的字体大小
ax.tick_params(axis='x', labelsize=ticksize)
ax.tick_params(axis='y', labelsize=ticksize)
# plt.ylim([-2.5, 2.5])
#set right and up axis off
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#set left and bottom axis to 1 pt
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)

plt.tight_layout()

fig.savefig('Figures/Fig3b_2.pdf')

