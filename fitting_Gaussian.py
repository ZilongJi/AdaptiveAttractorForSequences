import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import math

bm.set_platform('cpu')

class CANN2D(bp.dyn.NeuGroup):
  def __init__(self, length, tau=1, tau_v=144, m_0=1., k=8.1, a=0.2, A=10., J0=4., sigma_u=0., sigma_m=0.,
               z_min=-bm.pi, z_max=bm.pi, name=None):
    super(CANN2D, self).__init__(size=(length, length), name=name)

    # parameters
    self.length = length
    self.tau = tau  # The synaptic time constant
    self.tau_v = tau_v
    self.m = tau/tau_v*m_0
    self.k = k  # Degree of the rescaled inhibition
    self.a = a  # Half-width of the range of excitatory connections
    self.A = A  # Magnitude of the external input
    self.J0 = J0  # maximum connection value
    self.sigma_m = sigma_m
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
    self.center = bm.Variable(bm.zeros(2))

  def reset_state(self, batch_size=None):
    # variables
    self.r = bm.Variable(bm.zeros((self.length, self.length)))
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
    xcenter = bm.sum(self.r, axis=0)
    ycenter = bm.sum(self.r, axis=1)
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
                   + self.sigma_m * self.u * bm.random.normal(0, 1, (self.length, self.length)) * bm.sqrt(bm.get_dt() / self.tau_v)
    #self.v.value = self.v + (-self.v + self.m * self.r) / self.tau_v * bm.get_dt() \
                   #+ self.sigma_m * self.u * bm.random.normal(0, 1, (self.length, self.length)) * bm.sqrt(bm.get_dt() / self.tau_v)
    self.get_center()
    self.input[:] = 0.
# m = 1.13 boundary

def get_trace(mu, gamma, duration=10, a=0.2, tau=1, tau_v=20, sigma_u = 0.5, visulaize = False):
  def get_sigma_m(mu, gamma):
    m_0 = 1 - mu
    sigma_m = 2 * math.sqrt(np.pi) * m_0 * tau / tau_v * a * gamma
    print('sigma_m =', sigma_m, ', m_0 = ', m_0)
    return sigma_m, m_0

  sigma_m, m_0 = get_sigma_m(mu, gamma)

  cann = CANN2D(length=100, a=a, tau=tau, tau_v=tau_v, k=0.1, sigma_u=sigma_u, sigma_m=sigma_m, m_0=m_0)
  Iext, length = bp.inputs.section_input(
      values=[cann.get_stimulus_by_pos([1, 0.]), 0.],
      durations=[2., duration],
      return_length=True
  )

  runner = bp.DSRunner(cann,
                       inputs = ['input', Iext, 'iter'],
                       monitors = ['r', 'center'],
                       dt = 0.01,
                       numpy_mon_after_run=False)
  runner.run(length)
  fr = runner.mon.r
  return fr

mu = 1
gamma = 0.3
x = np.linspace(-np.pi, np.pi, 100)
y = np.linspace(-np.pi, np.pi, 100)
X, Y = np.meshgrid(x, y)

# 定义高斯函数
def gaussian(x, y, mean_x, mean_y, sigma_x, sigma_y):
    return np.exp(-((x - mean_x) ** 2 / (2 * sigma_x ** 2) + (y - mean_y) ** 2 / (2 * sigma_y ** 2)))

# 定义损失函数
def loss_function(params):
    mean_x, mean_y, sigma_x, sigma_y = params
    predicted = gaussian(X, Y, mean_x, mean_y, sigma_x, sigma_y).flatten()
    return np.mean((data - predicted) ** 2)

# 初始化参数
initial_params = np.array([1., 0.0, 0.2, 0.2])

Sigma = bm.linspace(0,1,10)
Loss = []
for sigma_u in Sigma:
  fr = get_trace(mu, gamma, 10, 0.2, 1, 1, sigma_u=sigma_u)
  data = fr[1100,:,:].flatten()
  # 最小化损失函数
  result = optimize.minimize(loss_function, initial_params, method='BFGS')
  estimated_mean_x, estimated_mean_y, estimated_sigma_x, estimated_sigma_y = result.x

  # 计算拟合指标（均方误差）
  fitting_loss = result.fun
  Loss.append(fitting_loss)
  # 打印结果
  print("Estimated Mean (x, y):", estimated_mean_x, estimated_mean_y)
  print("Estimated Sigma (x, y):", estimated_sigma_x, estimated_sigma_y)
  print("Fitting Loss (Mean Squared Error):", fitting_loss)
plt.plot(Loss)
plt.show()