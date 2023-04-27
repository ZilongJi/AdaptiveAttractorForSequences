import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import lax
from scipy import stats
import math
import time
import sys

bm.set_dt(1.)
bm.set_platform('cpu')
print(bp.__version__)


# sys.exit()


class CANN2D(bp.DynamicalSystemNS):
    def __init__(self, length=128, tau=10, tau_v=200, m_0=0.7, k=0.05, a=np.pi / 6, A=0.1, J0=1., sigma_u=0.,
                 sigma_v=0.5, z_min=-bm.pi, z_max=bm.pi, name=None):
        super(CANN2D, self).__init__(name=name)

        # parameters
        self.length = length
        self.tau = tau  # The synaptic time constant
        self.tau_v = tau_v
        self.m = tau / tau_v * m_0
        self.k = k  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0  # maximum connection value
        self.sigma_v = sigma_v
        self.sigma_u = sigma_u

        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = bm.linspace(z_min, z_max, length + 1)[0:-1]  # The encoded feature values
        self.rho = length / self.z_range  # The neural density
        self.dx = self.z_range / length  # The stimulus density

        # The connections
        self.conn_mat = self.make_conn()
        self.conn_fft = bm.fft.fft2(self.conn_mat)
        self.reset_state()

    def reset_state(self, batch_size=None):
        # variables
        self.center = bm.Variable(bm.zeros(2))
        self.r = bm.Variable(bm.zeros((self.length, self.length)))
        self.rm = bm.Variable(bm.zeros((self.length, self.length)))
        self.u = bm.Variable(bm.zeros((self.length, self.length)))
        self.v = bm.Variable(bm.zeros((self.length, self.length)))

    def show_conn(self):
        plt.imshow(bm.as_numpy(self.conn_mat))
        plt.colorbar()
        plt.show()

    def dist(self, d):
        v_size = bm.asarray([self.z_range, self.z_range])
        return bm.where(d > v_size / 2, v_size - d, d)

    def make_conn(self):
        x1, x2 = bm.meshgrid(self.x, self.x)
        value = bm.stack([x1.flatten(), x2.flatten()]).T
        d = self.dist(bm.abs(value[0] - value))
        d = bm.linalg.norm(d, axis=1)
        d = d.reshape((self.length, self.length))
        Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (2 * bm.pi * bm.square(self.a))
        return Jxx

    def get_stimulus_by_pos(self, pos):
        assert bm.size(pos) == 2
        x1, x2 = bm.meshgrid(self.x, self.x)
        value = bm.stack([x1.flatten(), x2.flatten()]).T
        d = self.dist(bm.abs(bm.asarray(pos) - value))
        d = bm.linalg.norm(d, axis=1)
        d = d.reshape((self.length, self.length))
        return self.A * bm.exp(-0.25 * bm.square(d / self.a))

    def get_center(self):
        r_c = self.r
        threshold = np.max(r_c) / 3.0
        r_c = bm.where(r_c > threshold, r_c, 0)
        xcenter = bm.sum(r_c, axis=0)
        ycenter = bm.sum(r_c, axis=1)
        exppos = bm.exp(1j * self.x)
        self.center[0] = bm.angle(bm.sum(exppos * xcenter))
        self.center[1] = bm.angle(bm.sum(exppos * ycenter))

    def update(self, inp):
        rfft = bm.fft.fft2(self.r)
        interaction = bm.real(bm.fft.ifft2(rfft * self.conn_fft))
        self.u.value = self.u + (-self.u + inp + interaction - self.v) / self.tau * bm.get_dt()
        self.v.value = self.v + (-self.v + self.m * self.u) / self.tau_v * bm.get_dt() \
                       + self.sigma_v * bm.random.normal(0, 1, (self.length, self.length)) * \
                       bm.sqrt(bm.get_dt() / self.tau_v)
        self.u.value = bm.where(self.u > 0, self.u, 0)
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2
        self.rm.value = bm.where(self.r > 0.01, self.r, 0)
        self.get_center()


def get_trace(duration=30000, beta=0.5, sample_rate=20, T_start=2000, T_sample=1000, visual=False):
    cann = CANN2D()
    Iext, length = bp.inputs.section_input(
        values=[cann.get_stimulus_by_pos([0., 0.]), 0.],
        durations=[500., duration],
        return_length=True
    )
    # 获取每个维度的长度并赋值给x和y
    x = len(Iext)  # 获取第一维长度
    y = len(Iext[0])  # 获取第二维长度

    # 创建与给定二维矩阵大小相同的矩阵
    T_gamma = 400
    xx = np.meshgrid(np.arange(x), np.arange(y), np.arange(y))[0]  # 重新生成 xx
    xx = xx.transpose((1, 0, 2))  # 将 xx 的第一维和第二维进行交换
    wave = beta * np.sin(xx * 2 * np.pi / T_gamma)
    Iext = bm.as_numpy(Iext) + wave

    def run_net(inp, ):  # 20 x size
        for i in range(sample_rate):
            cann.update(inp[i])
        return cann.center, cann.rm

    t0 = time.time()
    center_trace, fr = bm.for_loop(run_net, Iext.reshape(-1, sample_rate, cann.length, cann.length))
    print(time.time() - t0)

    center_trace = bm.as_numpy(center_trace)
    fr = bm.as_numpy(fr)

    stepx = np.diff(center_trace[:, 0])
    stepy = np.diff(center_trace[:, 1])
    stepx = np.where(stepx > np.pi, stepx - 2 * np.pi, stepx)
    stepy = np.where(stepy > np.pi, stepy - 2 * np.pi, stepy)
    stepx = np.where(stepx < -np.pi, stepx + 2 * np.pi, stepx)
    stepy = np.where(stepy < -np.pi, stepy + 2 * np.pi, stepy)
    step = np.sqrt(stepx ** 2 + stepy ** 2)
    # fr = runner.mon.r[T_start:T_start+T_sample:sample_rate, :, :]
    # fr = runner.mon.r[T_start:-1:sample_rate, :, :]
    mean_fr = np.mean(fr, axis=(1, 2))
    if visual == True:
        plt.hist(step, bins=60)
        plt.show()
        plt.plot(mean_fr[T_start:T_sample+T_start])
        plt.show()
        plt.plot(center_trace[T_start:, 0], center_trace[T_start:, 1])
        plt.show()
    return center_trace, step, mean_fr
