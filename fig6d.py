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
period = int(20)
phase_step = np.zeros(period)
phase_r = np.zeros(period)
phase = np.linspace(-np.pi, np.pi, period+1)[:-1]
for i in range(int(len(mean_fr) / period)):
    for j in range(period):
        phase_step[j] = phase_step[j] + step[i * period + j]
        phase_r[j] = phase_r[j] + mean_fr[i * period + j]

fig1, axs = plt.subplots(2, 1, figsize=(4, 6), sharex=True)
axs[0].plot(phase, phase_r)
down_sample = 1
axs[0].bar(phase[0:-1:down_sample], phase_r[0:-1:down_sample], width=np.pi/period)
axs[1].plot(phase[0:-1], phase_step[0:-1])
axs[1].bar(phase[0:-1:down_sample], phase_step[0:-1:down_sample], width=np.pi/period)
# axs[0].bar(phase[0:-1:down_sample], phase_step[0:-1:down_sample])
plt.show()
