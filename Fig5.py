import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from TwoD_gamma import CANN2D
import brainpy.math as bm
import brainpy as bp
import jax.numpy as jnp

duration = 30000
cann = CANN2D()
Iext, length = bp.inputs.section_input(
  values=[cann.get_stimulus_by_pos([0., 0.]), 0.],
  durations=[500., duration],
  dt = 1.,
  return_length=True
)
# 获取每个维度的长度并赋值给x和y
x = len(Iext)          # 获取第一维长度
y = len(Iext[0])       # 获取第二维长度

# 创建与给定二维矩阵大小相同的矩阵
T_gamma = 400
xx = np.meshgrid(np.arange(x), np.arange(y), np.arange(y))[0]  # 重新生成 xx
xx = xx.transpose((1, 0, 2))  # 将 xx 的第一维和第二维进行交换
wave = 0.5*np.sin(xx*2*np.pi/T_gamma)
# # 给新矩阵的元素赋值，要求与第一维索引的正弦函数成关系
# for i in range(x):
#     for j in range(y):
#         for k in range(y):
#             wave[i][j][k] = 0.5*np.sin(i*2*np.pi/T_gamma)
Iext = Iext + wave
print('start')
runner = bp.DSRunner(cann,
                   inputs = ['input', Iext, 'iter'],
                   monitors = ['r', 'center'],
                   dyn_vars = bm.random.DEFAULT,
                   dt = 1.,
                   jit = True,
                   numpy_mon_after_run=False,
                   progress_bar=True)
runner.run(length)
sample_rate = int(T_gamma/20)
center_trace = runner.mon.center[1000:-1:sample_rate,:]
stepx = np.diff(center_trace[:,0])
stepy = np.diff(center_trace[:,1])
stepx = bm.where(stepx > np.pi, stepx-2*np.pi, stepx)
stepy = bm.where(stepy > np.pi, stepy-2*np.pi, stepy)
stepx = bm.where(stepx < -np.pi, stepx+2*np.pi, stepx)
stepy = bm.where(stepy < -np.pi, stepy+2*np.pi, stepy)
step = np.sqrt(stepx**2+stepy**2)
plt.hist(step, bins = 30)
plt.show()
fr = runner.mon.r[2000:3000:sample_rate,:,:]
mean_fr = np.mean(fr,axis = (1,2))
# print(jnp.shape(mean_fr))
# print(type(mean_fr))
plt.plot(mean_fr)
plt.show()
# plt.scatter(center_trace[:,0], center_trace[:,1], c = np.linspace(1,0,center_trace.shape[0]), s = 1)
plt.plot(center_trace[:,0], center_trace[:,1])
plt.show()

