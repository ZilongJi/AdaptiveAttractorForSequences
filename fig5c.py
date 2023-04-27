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
cx = center_trace[w_start:(w_start+w_size),0]
cy = center_trace[w_start:(w_start+w_size),1]
st = step[w_start:(w_start+w_size)]
mr = mean_fr[w_start:(w_start+w_size)]

Peaks,_ = scipy.signal.find_peaks(-mr)
print(Peaks)
fig1, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
# 在第一个子图中绘制折线图
axs[0].plot(st)
axs[0].plot(st,'r.')
for peaks in Peaks:
    axs[0].plot([peaks, peaks],[0,np.max(st)],'k--', linewidth=1)
axs[1].plot(mr)
for peaks in Peaks:
    axs[1].plot([peaks, peaks],[0,np.max(mr)],'k--', linewidth=1)
plt.show()