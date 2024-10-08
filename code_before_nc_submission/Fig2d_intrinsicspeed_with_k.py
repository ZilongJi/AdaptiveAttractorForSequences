import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from cann import CANN1D

#%%
#set default ramndom seed for reproducibility
np.random.seed(0)
#set backend to cpu
bm.set_platform('cpu')


def intrinsic_speed(mbar=0, k=1.0, duration=5000):
    np.random.seed(0)
    cann = CANN1D(num=128, mbar=mbar, k=k, tau=3, tau_v=144)
    cann.reset_state()
    Iext, length = bp.inputs.section_input(
        values=[cann.get_stimulus_by_pos(0.), 0.],
        durations=[500., duration],
        return_length=True
    )
    noise = 0.02 * np.random.randn(len(Iext), cann.num)
    noise[5000:-1] = 0
    Iext = Iext + noise
    Iext = bm.as_numpy(Iext)
    runner = bp.DSRunner(cann,
                         inputs=('input', Iext, 'iter'),
                         monitors=['center'],
                         numpy_mon_after_run=False,
                         progress_bar=False)
    runner.run(length)
    center_trace = runner.mon.center[80000:-1]
    speed = bm.diff(center_trace, axis=0)
    mask = speed <= np.pi
    speed = speed[mask]
    v_int = bm.abs(np.mean(speed))*1e4
    # plt.plot(runner.mon.center)
    return v_int


num_p = 10
# monte_num = 20
allk = bm.linspace(4,8,num_p)

v_int = np.zeros(num_p,)
for i in range(num_p):
    v_int[i] = intrinsic_speed(mbar=2, k=allk[i], duration=15000)
    # print(v_int[i])
    print(allk[i])
    
    
 #%%   
fig, ax = plt.subplots(figsize=(2.4, 2.4),dpi=300)
#set parameters for the figure
labelsize = 10
ticksize = 8
custom_color = '#009FB9'
# set the linewidth of each axis
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1)

#plot Mbar v.s. v_int with scatters and lines
plt.plot(allk, v_int, color='k', linewidth=1, linestyle='--')
#add scatter points
plt.scatter(allk, v_int, s=30, c='#009FB9', marker='o', alpha=0.8, edgecolors='k')    

plt.xlabel(r'Global inhibition k', fontsize=labelsize)
plt.ylabel('$v_{int}$ (m/s)', fontsize=labelsize)

#set right and up axis  invisible
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
fig.savefig('Figures/Fig2d_intrinsicspeed_withk.pdf')

#%%
# bm.clear_buffer_memory()



# %%
