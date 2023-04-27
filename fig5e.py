import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import TwoD_gamma
import brainpy.math as bm
import brainpy as bp
import jax.numpy as jnp
import scipy

center_trace, step, mean_fr = TwoD_gamma.get_trace(duration=3e4, sample_rate=20, T_start=1000, visual=False)
w_start = 95
w_size = 160
w_step = 3
cx = center_trace[w_start:(w_start + w_size), 0]
cy = center_trace[w_start:(w_start + w_size), 1]
st = step[w_start:(w_start + w_size)]
mr = mean_fr[w_start:(w_start + w_size)]
period = 20
phase_step = np.zeros(period)
phase_r = np.zeros(period)
phase = np.linspace(-np.pi, np.pi, period+1)[:-1]
for i in range(int(len(mean_fr) / period)):
    for j in range(period):
        phase_step[j] = phase_step[j] + step[i * period + j]
        phase_r[j] = phase_r[j] + mean_fr[i * period + j]
phase_r = (phase_r-np.min(phase_r))/(np.max(phase_r)-np.min(phase_r))
phase_step = (phase_step-np.min(phase_step))/(np.max(phase_step)-np.min(phase_step))
# 首尾相连
phase = np.append(phase, phase[0])
phase_r = np.append(phase_r, phase_r[0])
phase_step = np.append(phase_step, phase_step[0])

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='polar')
ax.plot(phase, phase_r)
ax.plot(phase, phase_step)

plt.show()
