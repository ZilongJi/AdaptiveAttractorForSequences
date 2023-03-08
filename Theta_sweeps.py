import brainpy as bp
import brainpy.math as bm
import numpy as np
import jax
import seaborn as sns
import matplotlib.pyplot as plt
from cann_fft import CANN1D
bm.set_platform('cpu')

cann = CANN1D(tau=3, tau_v=144., num=128, mbar=155)
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
Iext = cann.get_stimulus_by_pos(position)

runner = bp.DSRunner(cann,
                     inputs=('input', Iext, 'iter'),
                     monitors=['u', 'v', 'r','centerI'])

runner.run(dur)
probe_num = int( 2*bm.pi / v_ext/dt)
plt.figure()
index = np.linspace(1, cann.num, cann.num)
pos = np.linspace(-np.pi,np.pi,cann.num+1)
pos = pos[0:-1]
fr = runner.mon.r.T[:,probe_num:-1]
plt.pcolormesh(time[probe_num:-1], pos, fr,cmap='jet')
cI = runner.mon.centerI[probe_num:-1]

plt.plot(time[probe_num:-1], cI, color='k')
plt.xlabel('time(ms)', fontsize=15)
plt.ylabel('Encoded Position', fontsize=15)
plt.ylim([-2.5, 2.5])
plt.colorbar()
plt.show()



# encoder = bp.encoding.PoissonEncoder()
# spike = encoder(fr.T*1e2)
# plt.figure()
# bp.visualize.raster_plot(time[probe_num:-1],spike)
# plt.show()