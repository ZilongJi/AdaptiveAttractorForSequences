
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import TwoD_gamma
import brainpy.math as bm
import brainpy as bp
import jax.numpy as jnp

center_trace, step, mean_fr = TwoD_gamma.get_trace(duration=3e4, sample_rate=20, T_start=1000, visual=False)
cx = center_trace[:,0]
cy = center_trace[:,1]
hist, bin_edges = np.histogram(step, bins=40)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
brownian_step = bm.random.normal(np.mean(step), np.var(step), (len(step), 1))
hist_brown, bin_edges_brown = np.histogram(brownian_step, bins=40)
bin_centers_brown = 0.5 * (bin_edges_brown[:-1] + bin_edges_brown[1:])
hist = hist/np.sum(hist)
hist_brown = hist_brown/np.sum(hist_brown)
fig1, axs = plt.subplots(figsize=(6, 4))
plt.clf()
plt.bar(bin_centers, hist, width=0.5*(bin_edges[1]-bin_edges[0]))
plt.plot(bin_centers, hist, 'blue')

plt.bar(bin_centers_brown, hist_brown, width=0.5*(bin_edges_brown[1]-bin_edges_brown[0]), color='brown')
plt.plot(bin_centers_brown, hist_brown, 'red')
plt.show()