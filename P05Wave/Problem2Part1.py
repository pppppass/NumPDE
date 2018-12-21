
# coding: utf-8

# In[1]:


import numpy
from matplotlib import pyplot
import models
import solvers


# In[2]:


def calc_norm2(vec, step):
    u, h = vec, step
    return numpy.sqrt(h * (numpy.linalg.norm(u[1:-1])**2 + 1.0/2.0 * (u[0]**2 + u[-1]**2)))


# In[3]:


def calc_log_line(start, end, intc, order):
    return [start, end], [intc, intc * (end / start)**order]


# In[4]:


ns = [3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024, 1448, 2048, 2896, 4096, 5793, 8192]
ts = [0.1, 0.9, 1.0, 1.9, 2.0, 4.0]
d = 4.0
nus = [0.25, 0.5, 0.75, 1.0]
nu_names = ["1 / 4", "1 / 2", "3 / 4", "1"]


# In[5]:


e = [[[] for nu in nus] for t in ts]


# In[6]:


for i, n in enumerate(ns):
    h = 1.0 / n
    x = numpy.linspace(0.0, 1.0, n+1)[:, None]
    for j, nu in enumerate(nus):
        tau = h * nu
        m = int(d / tau + 1.0e-5) + 2
        t = (tau * numpy.arange(m))[None, :]
        a = models.calc_coef(x, t)
        alpha = models.calc_alpha(t)
        g = models.calc_grad(t)
        f = models.get_dirich_flag()
        ctr = 0
        u = models.calc_ana_sol_1(x, 0.0)[:, 0]
        v_0 = models.calc_ana_sol_t_init_1(x)[:, 0]
        u_new = solvers.calc_first(u, v_0, h, tau, a[:, :2], alpha[:, :2], g[:, :2], f)
        for k, t_now in enumerate(ts):
            ctr_next = int(t_now / tau + 1.0e-5)
            for l in range(ctr, ctr_next):
                u, u_new = u_new, solvers.iter_expl(u, u_new, h, tau, a[:, i:i+2], alpha[:, i:i+2], g[:, i:i+2], f)
            ctr = ctr_next
            u_sol = (((ctr+1) * tau - t_now) / tau) * u + ((t_now - ctr * tau) / tau) * u_new
            u_ana = models.calc_ana_sol_1(x, t_now)[:, 0]
            e[k][j].append(calc_norm2(u_sol - u_ana, h))
            print("t = {} finished, ctr = {}".format(t_now, ctr))
        print("nu = {} finished".format(nu))
    print("n = {} finished".format(n))


# In[9]:


pyplot.figure(figsize=(8.0, 12.0))
for i, t in enumerate(ts):
    pyplot.subplot(3, 2, i+1)
    pyplot.title("$ t = {} $".format(t))
    for j, nu in enumerate(nus):
        pyplot.plot(ns, e[i][j], label="$ \\nu = {} $".format(nu_names[j]))
        pyplot.scatter(ns, e[i][j], s=2.0)
    pyplot.plot(*calc_log_line(3.0, 8192.0, 1.7e-2, -1.0), linewidth=0.5, color="black", label="Slope $-1$")
    if i in [0, 1, 2, 3]:
        pyplot.plot(*calc_log_line(3.0, 8192.0, 1.0e-2, -2.0), linewidth=0.5, color="black", linestyle="--", label="Slope $-2$")
    pyplot.plot([], linewidth=0.5, color="black", linestyle="--", label="Slope $-2$")
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("$n$")
    pyplot.ylabel("Error")
pyplot.legend()
pyplot.tight_layout()
pyplot.savefig("Figure09.pgf")
pyplot.show()
pyplot.close()

