import brainpy as bp
import brainpy.math as bm
import numpy as np
import jax
import seaborn as sns
import matplotlib.pyplot as plt
from cann import CANN1D

def delayed_track_m(mbar=0):
    import brainpy as bp
    from cann_fft import CANN1D
    cann = CANN1D(num=128, mbar=mbar, tau=3, tau_v=144)
    vbar = 1
    v_ext = cann.a / cann.tau_v * vbar
    dur = 0.1 * bm.pi / v_ext
    dt = bm.get_dt()
    num = int((dur // dt))
    position = np.zeros(num)
    position[0] = -np.pi/2
    for i in range(num)[1:]:
        position[i] = position[i - 1] + v_ext * dt
        if position[i] > np.pi:
            position[i] -= 2 * np.pi
    position = position.reshape((-1, 1))
    noise = 0.02 * np.random.randn(num, cann.num)
    Iext = cann.get_stimulus_by_pos(position)+noise
    runner = bp.DSRunner(cann,
                         inputs=('input', Iext, 'iter'),
                         monitors=['center', 'centerI'],
                         numpy_mon_after_run=False,
                         progress_bar=False)
    runner.run(dur)
    distance = runner.mon.center - runner.mon.centerI
    mean_dis = np.mean(distance[500:-1])
    return mean_dis

num_p = 10
monte_num = 20
mbar = bm.linspace(0,9.5,num_p)
dis = np.zeros((num_p, monte_num))
for monte in range(monte_num):
    dis[:,monte] = bp.running.jax_vectorize_map(delayed_track_m, [mbar], num_parallel=num_p)
    print(monte/monte_num)

mean_dis = np.mean(dis*1e3,axis=1)
std_dis = np.std(dis*1e3,axis=1)
# aspect_ratio = 4/3
ylen = 6
fig, ax = plt.subplots(figsize=(ylen, ylen))
ax.errorbar(mbar/5, -mean_dis, yerr = std_dis, fmt='o', color='blue', ecolor='black', capsize=5, capthick=2)
ax.fill_between(mbar/5, -mean_dis-std_dis, -mean_dis+std_dis, alpha=0.2, color='blue')
plt.scatter(mbar/5, -mean_dis)
plt.plot(mbar/5, -mean_dis)
plt.xlabel('Adaptation strength', fontsize=15)
plt.ylabel('Delayed distance (1e-3)', fontsize=15)
plt.tight_layout()
# 设置坐标轴的线条粗细
ax.spines['top'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
# 设置xtick和ytick的字体大小
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
fig.savefig('Figures/lag_distance_adaptation.png', dpi=300)
# plt.show()




num_p = 10
mbar = 0
cann = CANN1D(num=128, mbar=mbar, tau=3, tau_v=144)
vbar = bm.linspace(0.001,1,num_p)
monte_num = 40
dis = np.zeros((num_p, monte_num))

for ni in range(num_p):
    v_ext = cann.a / cann.tau_v * vbar[ni]
    dur = 0.1 * bm.pi / v_ext
    dt = bm.get_dt()
    num = (dur / dt).astype(int)
    position = np.zeros(num)
    position[0] = -np.pi/2
    for i in range(num)[1:]:
        position[i] = position[i - 1] + v_ext * dt
        if position[i] > np.pi:
            position[i] -= 2 * np.pi
    position = position.reshape((-1, 1))
    for monte in range(monte_num):
        noise = 0.02 * np.random.randn(num, cann.num)
        Iext = cann.get_stimulus_by_pos(position)+noise
        runner = bp.DSRunner(cann,
                             inputs=('input', Iext, 'iter'),
                             monitors=['center', 'centerI'],
                             numpy_mon_after_run=False,
                             progress_bar=False)
        runner.run(dur)
        distance = runner.mon.center - runner.mon.centerI
        dis[ni, monte] = np.mean(distance[500:-1])
        progress = ni*monte_num+monte
        print(progress/(num_p*monte_num))

std_dis = np.std(dis*1e3, axis=1)
mean_dis = np.mean(dis*1e3, axis=1)


vext = vbar*0.4/5*1e3
fig, ax = plt.subplots(figsize=(ylen, ylen))
ax.errorbar(vext, -mean_dis, yerr=std_dis, fmt='o', color='blue', ecolor='black', capsize=5, capthick=2)
ax.fill_between(vext, -mean_dis-std_dis, -mean_dis+std_dis, alpha=0.2, color='blue')
plt.scatter(vext, -mean_dis)
plt.plot(vext, -mean_dis)
plt.xlabel('Moving speed (rads/s)', fontsize=15)
plt.ylabel('Delayed distance (rads*1e-3)', fontsize=15)
plt.tight_layout()
fig.savefig('Figures/lag_distance_vext.png', dpi=300)
plt.show()




