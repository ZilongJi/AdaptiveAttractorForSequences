import brainpy as bp
import brainpy.math as bm
import numpy as np
import jax
import seaborn as sns
import matplotlib.pyplot as plt
from cann_fft import CANN1D
import scipy
bm.set_platform('cpu')

cann = CANN1D(tau=3, tau_v=144., num=128, mbar=150.5)
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
                     monitors=['u', 'v', 'r','center','centerI'])

runner.run(dur)
probe_num = int( 1.9*bm.pi / v_ext/dt)
time=time[probe_num:-1]
##Visualize theta sweeps
plt.figure()
index = np.linspace(1, cann.num, cann.num)
pos = np.linspace(-np.pi,np.pi,cann.num+1)
pos = pos[0:-1]
fr = runner.mon.r.T[:,probe_num:-1]
plt.pcolormesh(time, pos, fr, cmap='jet')
cI = runner.mon.centerI[probe_num:-1]
plt.plot(time, cI, color='k')
plt.xlabel('time(ms)', fontsize=15)
plt.ylabel('Encoded Position', fontsize=15)
plt.ylim([-2.5, 2.5])
plt.colorbar()
plt.show()
##Calculate theta phase
relative_pos = runner.mon.center[probe_num:-1] - runner.mon.centerI[probe_num:-1]
relative_pos = bm.where(relative_pos > np.pi, relative_pos - 2*np.pi,  relative_pos)
relative_pos = bm.where(relative_pos < -np.pi, relative_pos + 2*np.pi,  relative_pos)
relative_pos = np.squeeze(relative_pos)
Peaks,_ = scipy.signal.find_peaks(relative_pos, width=100)
Period = np.mean(np.diff(time[Peaks]))
phase = np.zeros(len(time))
for ti in range(len(time)):
    time_residual = time[ti]-time[Peaks]
    if np.max(time_residual)>0:
        time_residual[time_residual < 0] = 2*Period
        phase[ti] = np.min(time_residual)/Period*2*np.pi
    else:
        phase[ti] = (Period+np.max(time_residual)) / Period * 2 * np.pi





# encoder = bp.encoding.PoissonEncoder()
# spike = encoder(fr.T*1e2)
# plt.figure()
# bp.visualize.raster_plot(time[probe_num:-1],spike)
# plt.show()