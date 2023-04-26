import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from cann import CANN1D

#set default ramndom seed for reproducibility
np.random.seed(0)
#set backend to cpu
bm.set_platform('cpu')


num_p = 10
mbar = 0
cann = CANN1D(num=64, mbar=mbar, tau=3, tau_v=144)
vbar = bm.linspace(0.01,4,num_p)
monte_num = 40
dis = np.zeros((num_p, monte_num))

for ni in range(num_p):
    v_ext = cann.a / cann.tau_v * vbar[ni]
    dur = 0.2 * np.pi / v_ext
    dt = bm.get_dt()
    num = (dur / dt).astype(int)
    position = np.linspace(0,dur*v_ext,num)
    position = position.reshape((-1, 1))
    for monte in range(monte_num):
        noise = 0.06 * np.random.randn(num, cann.num)
        Iext = cann.get_stimulus_by_pos(position)+noise
        cann.reset_state()
        # looper = bp.LoopOverTime(cann)
        # out = looper(Iext)
        runner = bp.DSRunner(cann,
                             inputs=('input', Iext, 'iter'),
                             monitors=['center', 'centerI'],
                             numpy_mon_after_run=False,
                             progress_bar=False)
        runner.run(dur)
        distance = runner.mon.center - runner.mon.centerI
        # distance = out.center - out.centerI
        dis[ni, monte] = np.mean(distance[500:-1])
        progress = ni*monte_num+monte
        # cann.reset_state()
        print(progress/(num_p*monte_num))
        # bm.clear_buffer_memory()
std_dis = np.std(dis*1e2, axis=1)
mean_dis = np.mean(dis*1e2, axis=1)

#%%
vext = vbar*0.4/144*1e3
fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
#set parameters
labelsize = 18
ticksize = 14
custom_color = '#009FB9'
# set the linewidth of each axis
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1)

ax.errorbar(vext, -mean_dis, yerr=std_dis, fmt='o', color=custom_color, ecolor='gray', capsize=4, capthick=1)
ax.fill_between(vext, -mean_dis-std_dis, -mean_dis+std_dis, alpha=0.2, color=custom_color)
plt.scatter(vext, -mean_dis, color=custom_color, s=50, edgecolors='k')
plt.plot(vext, -mean_dis, color='k', linestyle=':', linewidth=2)

plt.xlabel(r'Moving speed $v_{ext}$ (cm/s)', fontsize=labelsize)
plt.ylabel('Lag distance (cm)', fontsize=labelsize)

# set the xticks and yticks
# yticks = np.array([0,0.5,1,1.5])
xticks = np.linspace(0,10,5)
# xticks = np.array([0,0.5,1,1.5,2,])
# ax.set_xticks(xticks)
# ax.set_yticks(yticks)
#chnage xtick labels by times 100
xticklabels = [str(int(xtick*100)) for xtick in xticks]
ax.set_xticklabels(xticklabels, fontsize=ticksize)

ax.tick_params(axis='x', labelsize=ticksize)
ax.tick_params(axis='y', labelsize=ticksize)

#set right and up axis  invisible
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()
fig.savefig('Figures/Fig2d.pdf', dpi=300)






# %%
