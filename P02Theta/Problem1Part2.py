
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
import models
import exts


# In[2]:


w, d = 1.0, 2.0
n = 100
h = w / n
theta = 0.0


# In[3]:


mu_list = 0.5 * 10.0 / numpy.log(1.0 + numpy.linspace(0.0, 2.0, 11)[1:])
p_name = ["{:.1f}".format(p) for p in numpy.linspace(0.0, 2.0, 11)[1:]]


# In[4]:


norm_2_mask = numpy.ones(n+1)
norm_2_mask[[0, -1]] /= numpy.sqrt(2)


# In[5]:


t_chk = numpy.linspace(0.0, 2.0, 101)


# In[6]:


rt = [{}, {}]


# In[7]:


for mu_ in mu_list:
    
    rt[0][mu_] = []
    rt[1][mu_] = []
    
    m = int(d / mu_ / h**2 - 1.0e-5) + 1
    tau = d / m

    x = numpy.linspace(0.0, w, n+1)[:, None]
    u = numpy.zeros((n+1, m+1))
    u[:, 0] = models.calc_ana_sol_2(x, 0.0)[:, 0]

    print("m = {} started".format(m))
    start = time.time()

    t = numpy.linspace(0.0, d, m+1)[None, :]

    a = models.calc_coef_2(x, t)
    f = models.calc_sour_2(x, t)
    alpha = models.calc_alpha_2(t)
    g = models.calc_grad_2(t)

    exts.para_theta_ghost_full_wrapper(n, m, w, d, a, f, alpha, g, theta, u, m+1)

    for t_ in t_chk:
        m_ = int(t_ / tau)
        u_m_sol = u[:, m_]
        u_m_ana = models.calc_ana_sol_2(x, t[0, m_])[:, 0]
        rt[0][mu_].append((numpy.linalg.norm((u_m_sol - u_m_ana) * norm_2_mask, 2.0) * numpy.sqrt(h)))
        rt[1][mu_].append(t[0, m_])

    del(u_m_sol)
    del(u_m_ana)
    del(u)
    del(f)
    del(a)


# In[8]:


with shelve.open("Result") as db:
    db[str((1, 2, "mu"))] = mu_list
    db[str((1, 2, "p", "name"))] = p_name
    db[str((1, 2, "norm2"))] = rt[0]
    db[str((1, 2, "t_m"))] = rt[1]

