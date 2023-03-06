import brainpy as bp
import brainpy.math as bm
import jax

bm.set_platform('cpu')
class CANN1D(bp.NeuGroup):
  def __init__(self, num, tau=1., tau_v=50., k=5., a=0.3, A=0.12, J0=1.,
               z_min=-bm.pi, z_max=bm.pi, mbar=150):
    super(CANN1D, self).__init__(size=num)

    # parameters
    self.tau = tau  # The synaptic time constant
    self.tau_v = tau_v
    self.k = k  # Degree of the rescaled inhibition
    self.a = a  # Half-width of the range of excitatory connections
    self.A = A  # Magnitude of the external input
    self.J0 = J0  # maximum connection value
    self.m = mbar*tau/tau_v

    # feature space
    self.z_min = z_min
    self.z_max = z_max
    self.z_range = z_max - z_min
    self.x = bm.linspace(z_min, z_max, num)  # The encoded feature values
    self.rho = num / self.z_range  # The neural density
    self.dx = self.z_range / num  # The stimulus density

    # The connection matrix
    # self.conn_mat = self.make_conn()
    conn_mat = self.make_conn()
    self.conn_fft = bm.fft.fft(conn_mat)

    # variables
    self.r = bm.Variable(bm.zeros(num))
    self.u = bm.Variable(bm.zeros(num))
    self.v = bm.Variable(bm.zeros(num))
    self.input = bm.Variable(bm.zeros(num))

  def dist(self, d):
    d = bm.remainder(d, self.z_range)
    d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
    return d

  def make_conn(self):
    d = self.dist(bm.abs(self.x[0] - self.x))
    Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
    return Jxx

  def get_stimulus_by_pos(self, pos):
    return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - pos) / self.a))

  def update(self, tdi):
    r1 = bm.square(self.u)
    r2 = 1.0 + self.k * bm.sum(r1)
    self.r.value = r1 / r2
    # r = jax.vmap(bm.fft.fft)(self.r)
    r = bm.fft.fft(self.r)
    Irec = bm.real(bm.fft.ifft(r * self.conn_fft))
    # Irec = bm.dot(self.conn_mat, self.r)
    self.u.value = self.u + (-self.u + Irec + self.input - self.v) / self.tau * tdi.dt
    self.v.value = self.v + (-self.v + self.m * self.u) / self.tau_v * tdi.dt
    self.input[:] = 0.

cann = CANN1D(num=512)
dur1, dur2, dur3 = 100., 2000., 500.
num1 = int(dur1 / bm.get_dt())
num2 = int(dur2 / bm.get_dt())
num3 = int(dur3 / bm.get_dt())
position = bm.zeros(num1 + num2 + num3)
final_pos = cann.a / cann.tau_v * 0.6 * dur2
position[num1: num1 + num2] = bm.linspace(0., final_pos, num2)
position[num1 + num2:] = final_pos
position = position.reshape((-1, 1))
Iext = cann.get_stimulus_by_pos(position)

runner = bp.DSRunner(cann,
                     inputs=('input', Iext, 'iter'),
                     monitors=['u', 'v'])
runner.run(dur1 + dur2 + dur3)
_ = bp.visualize.animate_1D(
  dynamical_vars=[
    {'ys': runner.mon.u, 'xs': cann.x, 'legend': 'u'},
    {'ys': runner.mon.v, 'xs': cann.x, 'legend': 'v'},
    {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}
  ],
  frame_step=30,
  frame_delay=5,
  show=True
)