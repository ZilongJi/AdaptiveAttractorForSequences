import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import lax
from scipy import stats
import math

bm.set_platform('cpu')

class CANN2D(bp.dyn.NeuGroup):
  def __init__(self, length=64, tau=10, tau_v=200, m_0=0.7, k=0.05, a=np.pi/6, A=10., J0=1., sigma_u=0., sigma_v=0.5,
               z_min=-bm.pi, z_max=bm.pi, name=None):
    super(CANN2D, self).__init__(size=(length, length), name=name)

    # parameters
    self.length = length
    self.tau = tau  # The synaptic time constant
    self.tau_v = tau_v
    self.m = tau/tau_v*m_0
    self.k = k*(length**2)  # Degree of the rescaled inhibition
    self.a = a  # Half-width of the range of excitatory connections
    self.A = A  # Magnitude of the external input
    self.J0 = J0  # maximum connection value
    self.sigma_v = sigma_v
    self.sigma_u = sigma_u

    # feature space
    self.z_min = z_min
    self.z_max = z_max
    self.z_range = z_max - z_min
    self.x = bm.linspace(z_min, z_max, length + 1)[0:-1] # The encoded feature values
    self.rho = length / self.z_range  # The neural density
    self.dx = self.z_range / length  # The stimulus density

    # The connections
    self.conn_mat = self.make_conn()
    self.conn_fft = bm.fft.fft2(self.conn_mat)
    self.reset_state()
    self.center = bm.zeros(2)

  def reset_state(self, batch_size=None):
    # variables
    self.r = bm.Variable(bm.zeros((self.length, self.length)))
    self.Mr = 0
    self.u = bm.Variable(bm.zeros((self.length, self.length)))
    self.v = bm.Variable(bm.zeros((self.length, self.length)))
    self.input = bm.Variable(bm.zeros((self.length, self.length)))

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
    Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
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
    self.center[0] = bm.angle(bm.sum(exppos*xcenter))
    self.center[1] = bm.angle(bm.sum(exppos*ycenter))

  def update(self, tdi):
    r1 = bm.square(self.u)
    r2 = 1.0 + self.k * bm.sum(r1)
    self.r.value = r1 / r2
    r = bm.fft.fft2(self.r)
    interaction = bm.real(bm.fft.ifft2(r * self.conn_fft))
    self.u.value = self.u + (-self.u + self.input + interaction - self.v) / self.tau * bm.get_dt() \
                   + self.sigma_u * bm.random.normal(0, 1, (self.length, self.length)) * bm.sqrt(bm.get_dt() / self.tau)
    self.v.value = self.v + (-self.v + self.m * self.u) / self.tau_v * bm.get_dt() \
                   + self.sigma_v * bm.random.normal(0, 1, (self.length, self.length)) * bm.sqrt(bm.get_dt() / self.tau_v)
    self.get_center()
    self.input[:] = 0.
