
# coding: utf-8

# In[1]:


import numpy
from matplotlib import pyplot
import models
import solvers


# In[2]:


n = 100
h = 1.0 / n
d = 2.0
ts = [0.0, 0.1, 0.2, 0.8, 0.9, 1.0, 1.1, 1.2, 1.8, 1.9, 2.0]
m = 400
tau = d / m


# In[3]:


x = numpy.linspace(0.0, 1.0, n+1)[:, None]
t = numpy.linspace(0.0, d, m+1)[None, :]


# In[4]:


u_ana = models.calc_ana_sol_2(x, t)


# In[5]:


pyplot.figure(figsize=(6.0, 4.0))
for t_now in ts:
    m_now = int(t_now / tau + 1.0e-5)
    pyplot.plot(x, u_ana[:, m_now], label="$ t = {:.1f} $".format(t_now))
pyplot.xlabel("$x$")
pyplot.ylabel("$u$")
pyplot.savefig("Figure03.pgf")
pyplot.legend()
pyplot.show()


# In[6]:


a = models.calc_coef(x, t)
alpha = models.calc_alpha(t)
g = models.calc_grad(t)
f = models.get_dirich_flag()


# In[7]:


u_0 = models.calc_ana_sol_2(x, 0.0)
v_0 = models.calc_ana_sol_t_init_2(x)


# In[8]:


u_sol = solvers.driver_expl(m, u_0, v_0, h, tau, a, alpha, g, f)


# In[9]:


pyplot.figure(figsize=(6.0, 4.0))
for t_now in ts:
    m_now = int(t_now / tau + 1.0e-5)
    pyplot.plot(x, u_sol[:, m_now], label="$ t = {:.1f} $".format(t_now))
pyplot.xlabel("$x$")
pyplot.ylabel("$U$")
pyplot.savefig("Figure04.pgf")
pyplot.legend()
pyplot.show()

