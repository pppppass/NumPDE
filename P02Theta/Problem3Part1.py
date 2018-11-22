
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
import models
import exts


# In[2]:


solver_list = ["full", "direct", "half"]
solver_name = ["Ghost full", "Direct", "Ghost half"]


# In[3]:


w, d = 1.0, 1000.0
n = 100
h = w / n


# In[4]:


theta = 1.0 / 2.0
m_list = numpy.logspace(0.0, 5.0, 41) * d


# In[5]:


t_chk = [1.0, 10.0, 100.0, 1000.0]
t_name = [1, 10, 100, 1000]


# In[6]:


norm_2_mask = numpy.ones(n+1)
norm_2_mask[[0, -1]] /= numpy.sqrt(2)


# In[7]:


rt = [{}, {}]


# In[8]:


for t in t_chk:
    rt[0][(t, "full")] = []
    rt[1][(t, "full")] = []

for m in m_list:

    m = int(m + 0.5)
    p =  (n * m - 1) // 150000000 + 1
    tau = d / m

    x = numpy.linspace(0.0, w, n+1)[:, None]
    ldu = m // p + 2
    u = numpy.zeros((n+1, ldu))
    u[:, 0] = models.calc_ana_sol_3(x, 0.0)[:, 0]

    t_ctr = 0

    print("m = {} started, {} chunks".format(m, p))

    for k in range(p):

        m_l, m_u = (k * m) // p, ((k+1) * m) // p
        t = numpy.linspace(tau * m_l, tau * m_u, m_u - m_l + 1)[None, :]

        a = models.calc_coef_3(x, t)
        f = models.calc_sour_3(x, t)
        alpha = models.calc_alpha_3(t)
        g = models.calc_grad_3(t)

        exts.para_theta_ghost_full_wrapper(n, m_u - m_l, w, tau * (m_u - m_l), a, f, alpha, g, theta, u, ldu)

        while t_ctr < len(t_chk):
            t_ = t_chk[t_ctr]
            if t_ > tau * m_u + 1.0e-10:
                break
            m_ = int(t_ / tau)
            u_m_sol = u[:, m_ - m_l]
            u_m_ana = models.calc_ana_sol_3(x, t[0, m_ - m_l])[:, 0]
            rt[1][(t_, "full")].append(numpy.linalg.norm(u_m_sol - u_m_ana, numpy.infty))
            rt[0][(t_, "full")].append(numpy.linalg.norm((u_m_sol - u_m_ana) * norm_2_mask, 2.0) * numpy.sqrt(h))
            t_ctr += 1

        u[:, 0] = u[:, m_u - m_l]

    del(u_m_sol)
    del(u_m_ana)
    del(u)
    del(f)
    del(a)


# In[9]:


for t in t_chk:
    rt[0][(t, "direct")] = []
    rt[1][(t, "direct")] = []

for m in m_list:

    m = int(m + 0.5)
    p =  (n * m - 1) // 150000000 + 1
    tau = d / m

    x = numpy.linspace(0.0, w, n+1)[:, None]
    ldu = m // p + 2
    u = numpy.zeros((n+1, ldu))
    u[:, 0] = models.calc_ana_sol_3(x, 0.0)[:, 0]

    t_ctr = 0

    print("m = {} started, {} chunks".format(m, p))

    for k in range(p):

        m_l, m_u = (k * m) // p, ((k+1) * m) // p
        t = numpy.linspace(tau * m_l, tau * m_u, m_u - m_l + 1)[None, :]

        a = models.calc_coef_3(x, t)
        f = models.calc_sour_3(x, t)
        alpha = models.calc_alpha_3(t)
        g = models.calc_grad_3(t)

        exts.para_theta_direct_wrapper(n, m_u - m_l, w, tau * (m_u - m_l), a, f, alpha, g, theta, u, ldu)

        while t_ctr < len(t_chk):
            t_ = t_chk[t_ctr]
            if t_ > tau * m_u + 1.0e-10:
                break
            m_ = int(t_ / tau)
            u_m_sol = u[:, m_ - m_l]
            u_m_ana = models.calc_ana_sol_3(x, t[0, m_ - m_l])[:, 0]
            rt[1][(t_, "direct")].append(numpy.linalg.norm(u_m_sol - u_m_ana, numpy.infty))
            rt[0][(t_, "direct")].append(numpy.linalg.norm((u_m_sol - u_m_ana) * norm_2_mask, 2.0) * numpy.sqrt(h))
            t_ctr += 1

        u[:, 0] = u[:, m_u - m_l]

    del(u_m_sol)
    del(u_m_ana)
    del(u)
    del(f)
    del(a)


# In[10]:


for t in t_chk:
    rt[0][(t, "half")] = []
    rt[1][(t, "half")] = []

for m in m_list:

    m = int(m + 0.5)
    p =  (n * m - 1) // 150000000 + 1
    tau = d / m

    x = numpy.linspace(w / 2.0 / n, w - w / 2.0 / n, n)[:, None]
    ldu = m // p + 2
    u = numpy.zeros((n, ldu))
    u[:, 0] = models.calc_ana_sol_3(x, 0.0)[:, 0]

    t_ctr = 0

    print("m = {} started, {} chunks".format(m, p))

    for k in range(p):

        m_l, m_u = (k * m) // p, ((k+1) * m) // p
        t = numpy.linspace(tau * m_l, tau * m_u, m_u - m_l + 1)[None, :]

        a = models.calc_coef_3(x, t)
        f = models.calc_sour_3(x, t)
        alpha = models.calc_alpha_3(t)
        g = models.calc_grad_3(t)

        exts.para_theta_ghost_half_wrapper(n, m_u - m_l, w, tau * (m_u - m_l), a, f, alpha, g, theta, u, ldu)

        while t_ctr < len(t_chk):
            t_ = t_chk[t_ctr]
            if t_ > tau * m_u + 1.0e-10:
                break
            m_ = int(t_ / tau)
            u_m_sol = u[:, m_ - m_l]
            u_m_ana = models.calc_ana_sol_3(x, t[0, m_ - m_l])[:, 0]
            rt[1][(t_, "half")].append(numpy.linalg.norm(u_m_sol - u_m_ana, numpy.infty))
            rt[0][(t_, "half")].append(numpy.linalg.norm(u_m_sol - u_m_ana, 2.0) * numpy.sqrt(h))
            t_ctr += 1

        u[:, 0] = u[:, m_u - m_l]

    del(u_m_sol)
    del(u_m_ana)
    del(u)
    del(f)
    del(a)


# In[14]:


with shelve.open("Result") as db:
    db[str((3, 1, "t"))] = t_chk
    db[str((3, 1, "t", "name"))] = t_name
    db[str((3, 1, "solver"))] = solver_list
    db[str((3, 1, "solver", "name"))] = solver_name
    db[str((3, 1, "m"))] = m_list
    db[str((3, 1, "norm2"))] = rt[0]
    db[str((3, 1, "normi"))] = rt[1]

