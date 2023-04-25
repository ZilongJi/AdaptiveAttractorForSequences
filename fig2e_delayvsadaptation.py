import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from cann import CANN1D
#set default ramndom seed for reproducibility
np.random.seed(0)
#set backend to cpu
bm.set_platform('cpu')


def delayed_track_m(mbar=0):
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

mean_dis = np.mean(dis*1e2,axis=1)
std_dis = np.std(dis*1e2,axis=1)
# aspect_ratio = 4/3
ylen = 6

#plot the figure
fig, ax = plt.subplots(figsize=(4, 3),dpi=300)
#set parameters for the figure
labelsize = 18
ticksize = 14
custom_color = '#009FB9'
# set the linewidth of each axis
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1)

# errorbar
ax.errorbar(mbar*3/144, -mean_dis, yerr = std_dis, fmt='o', color=custom_color, ecolor='gray', capsize=4, capthick=1)
ax.fill_between(mbar*3/144, -mean_dis-std_dis, -mean_dis+std_dis, alpha=0.2, color=custom_color)
plt.scatter(mbar*3/144, -mean_dis, color=custom_color, s=50, edgecolors='k')
plt.plot(mbar*3/144, -mean_dis, color='k', linestyle=':', linewidth=2)

plt.xlabel(r'Adaptation strength $m$', fontsize=labelsize)
plt.ylabel('Lag distance (cm)', fontsize=labelsize)

# ste the xticks and yticks
xticks = np.array([0,0.1,0.2])
yticks = np.array([0,0.5,1,1.5])

ax.set_xticks(xticks)
ax.set_yticks(yticks)

ax.tick_params(axis='x', labelsize=ticksize)
ax.tick_params(axis='y', labelsize=ticksize)

#set right and up axis  invisible
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()

fig.savefig('Figures/Fig2e.pdf')

bm.clear_buffer_memory()




