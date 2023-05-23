import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from cann import CANN1D
import time
#set default ramndom seed for reproducibility
np.random.seed(0)
#set backend to cpu
bm.set_platform('cpu')


def intrinsic_speed(mbar=0, duration=5000):
    np.random.seed(0)
    cann = CANN1D(num=128, mbar=mbar, tau=3, tau_v=144)
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


num_p = 26
# monte_num = 20
Mbar = bm.linspace(0,2.5,num_p)
v_int = np.zeros(num_p,)
for i in range(num_p):
    v_int[i] = intrinsic_speed(mbar=Mbar[i], duration=15000)
    # print(v_int[i])
    print(i/num_p)
fig, ax = plt.subplots(figsize=(4.5, 3),dpi=200)
#set parameters for the figure
labelsize = 18
ticksize = 14
custom_color = '#009FB9'
# set the linewidth of each axis
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1)

plt.plot(Mbar*3/144, v_int, color='k', linewidth=2)

plt.xlabel(r'Adaptation strength $m$', fontsize=labelsize)
plt.ylabel('v_int (m/s)', fontsize=labelsize)

# ste the xticks and yticks
# xticks = np.array([0,0.1,0.2])
# yticks = np.array([0,0.5,1,1.5])

# ax.set_xticks(xticks)
# ax.set_yticks(yticks)

ax.tick_params(axis='x', labelsize=ticksize)
ax.tick_params(axis='y', labelsize=ticksize)

#set right and up axis  invisible
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()
fig.savefig('Figures/v_int.pdf')

bm.clear_buffer_memory()




