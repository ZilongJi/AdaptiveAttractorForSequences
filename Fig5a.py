import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import TwoD_gamma
import brainpy.math as bm
import brainpy as bp
import jax.numpy as jnp

center_trace, step, mean_fr = TwoD_gamma.get_trace(duration=3e4, sample_rate=20, T_start=1000, visual=False)
w_start = 95
w_size = 160
w_step = 3
cx = center_trace[w_start:(w_start+w_size),0]
cy = center_trace[w_start:(w_start+w_size),1]
st = step[w_start:(w_start+w_size)]
mr = mean_fr[w_start:(w_start+w_size)]

fig1, axs = plt.subplots(3, 1, figsize=(6, 4))
# 在第一个子图中绘制折线图
axs[0].plot(cx)
axs[1].plot(cy)
axs[2].plot(st)


x = cx[0:-1:w_step]
y = cy[0:-1:w_step]
fig2, ax = plt.subplots(figsize=(4, 3))
size = 128
sigma = 0.02
xx = np.linspace(np.min(x)-0.1,np.max(cx)+0.1,size)
yy = np.linspace(np.min(y)-0.1,np.max(cy)+0.1,size)
X,Y = np.meshgrid(xx,yy)
Z = np.zeros((size,size))
for i in range(x.shape[0]):
    Z += np.exp((-(X-x[i])**2-(Y-y[i])**2)/(2*sigma**2))
ax.plot(x, y, 'blue',linewidth=0.5)
ax.contourf(X, Y, Z, alpha=1, levels=100,cmap='inferno')
# plt.show()

Z = np.zeros((size,size))
for i in range(x.shape[0]):
    if i/5 == np.floor(i/5):
        Z = np.exp((-(X-x[i])**2-(Y-y[i])**2)/(2*sigma**2))
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.contourf(X, Y, Z, alpha=1, levels=100,cmap='inferno')

plt.show()