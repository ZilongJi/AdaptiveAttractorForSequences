import brainpy as bp
import brainpy.math as bm
import numpy as np
import jax
import seaborn as sns
import matplotlib.pyplot as plt
from cann_fft import CANN1D
bm.set_platform('cpu')

cann = CANN1D(num=512,mbar = 0)
v_ext = cann.a / cann.tau_v * 0.5
dur = 2*bm.pi/v_ext
dt = bm.get_dt()
num = int(dur / dt)
time = np.linspace(0,dur,num)
final_pos = v_ext * dur
position = np.zeros(num)
position[0] = -np.pi
for i in range(num)[1:]:
    position[i] = position[i-1] + v_ext*dt
    if position[i] > np.pi:
        position[i] -= 2*np.pi

position = position.reshape((-1, 1))
Iext = cann.get_stimulus_by_pos(position)

runner = bp.DSRunner(cann,
                     inputs=('input', Iext, 'iter'),
                     monitors=['u', 'v', 'r'])

plt.figure()
index = np.linspace(1,cann.num,cann.num)
plt.pcolormesh(index,position,runner.mon.u)
plt.xlabel('Neuron index',fontsize = 15)
plt.ylabel('Position',fontsize = 15)
plt.colorbar
plt.show()