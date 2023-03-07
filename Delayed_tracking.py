import brainpy as bp
import brainpy.math as bm
import numpy as np
import jax
import seaborn as sns
import matplotlib.pyplot as plt
from cann_fft import CANN1D

bm.set_platform('cpu')



def delayed_track_m(mbar):
    import brainpy as bp
    from cann_fft import CANN1D
    cann = CANN1D(num=512, mbar=mbar, tau_v=5)
    vbar = 0.02
    v_ext = cann.a / cann.tau_v * vbar
    dur = 0.1 * bm.pi / v_ext
    dt = bm.get_dt()
    num = int(dur / dt)
    position = np.zeros(num)
    position[0] = -np.pi
    for i in range(num)[1:]:
        position[i] = position[i - 1] + v_ext * dt
        if position[i] > np.pi:
            position[i] -= 2 * np.pi
    position = position.reshape((-1, 1))
    Iext = cann.get_stimulus_by_pos(position)
    runner = bp.DSRunner(cann,
                         inputs=('input', Iext, 'iter'),
                         monitors=['center', 'centerI'],
                         numpy_mon_after_run=False)
    runner.run(dur)
    distance = runner.mon.center - runner.mon.centerI
    mean_dis = np.mean(distance[500:-1])
    return mean_dis

num_p = 10
mbar = bm.linspace(0,1,num_p)
mean_dis = bp.running.jax_vectorize_map(delayed_track_m, [mbar], num_parallel=num_p)
plt.figure()
plt.plot(mbar/5, -mean_dis*1e3)
plt.xlabel('Adaptation strength', fontsize=15)
plt.ylabel('Delayed distance (1e-3)', fontsize=15)
plt.show()

mbar = 0
cann = CANN1D(num=512, mbar=mbar, tau_v=5)
vbar = bm.linspace(0.001,0.02,num_p)
mean_dis = np.zeros(num_p)
for ni in range(num_p):
    v_ext = cann.a / cann.tau_v * vbar[ni]
    dur = 0.1 * bm.pi / v_ext
    dt = bm.get_dt()
    num = (dur / dt).astype(int)
    position = np.zeros(num)
    position[0] = -np.pi
    for i in range(num)[1:]:
        position[i] = position[i - 1] + v_ext * dt
        if position[i] > np.pi:
            position[i] -= 2 * np.pi
    position = position.reshape((-1, 1))
    Iext = cann.get_stimulus_by_pos(position)
    runner = bp.DSRunner(cann,
                         inputs=('input', Iext, 'iter'),
                         monitors=['center', 'centerI'],
                         numpy_mon_after_run=False)
    runner.run(dur)
    distance = runner.mon.center - runner.mon.centerI
    mean_dis[ni] = np.mean(distance[500:-1])


plt.figure()
plt.plot(vbar*0.4/5*1e3, -mean_dis*1e3)
plt.xlabel('Moving speed (1e-3)', fontsize=15)
plt.ylabel('Delayed distance (1e-3)', fontsize=15)
plt.show()

num_p = 10
mbar = 0.95
cann = CANN1D(num=512, mbar=mbar, tau_v=5)
vbar = bm.linspace(0.001,0.02,num_p)
mean_dis = np.zeros(num_p)
for ni in range(num_p):
    v_ext = cann.a / cann.tau_v * vbar[ni]
    dur = 0.1 * bm.pi / v_ext
    dt = bm.get_dt()
    num = (dur / dt).astype(int)
    position = np.zeros(num)
    position[0] = -np.pi
    for i in range(num)[1:]:
        position[i] = position[i - 1] + v_ext * dt
        if position[i] > np.pi:
            position[i] -= 2 * np.pi
    position = position.reshape((-1, 1))
    Iext = cann.get_stimulus_by_pos(position)
    runner = bp.DSRunner(cann,
                         inputs=('input', Iext, 'iter'),
                         monitors=['center', 'centerI'],
                         numpy_mon_after_run=False)
    runner.run(dur)
    distance = runner.mon.center - runner.mon.centerI
    mean_dis[ni] = np.mean(distance[500:-1])


plt.figure()
plt.plot(vbar*0.4/5*1e3, -mean_dis*1e3)
plt.axis([0,max(vbar)*0.4/5*1e3,-0.1,0.1])
plt.xlabel('Moving speed (1e-3)', fontsize=15)
plt.ylabel('Delayed distance (1e-3)', fontsize=15)
plt.tight_layout()
plt.show()
