
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
import models
import exts


# In[2]:


w, d = 1.0, 10.0
mu_name = [" \\tau / h = 4 ", " \\tau / h^2 = 1 / 15 "]
n_list = [
    [3, 4, 6, 8, 12, 17, 25, 35, 50, 70, 100, 141, 200, 282, 400, 565, 800, 1131, 1600, 2262, 3200, 4525, 6400, 9050, 12800, 18101, 25600, 36203],
    [3, 4, 6, 8, 12, 17, 25, 35, 50, 70, 100, 141, 200, 282, 400],
]
theta_list = [0.0, 1.0 / 2.0, 1.0]
theta_name = ["0", " 1 / 2 ", "1"]
solver_list = ["full", "direct", "half"]
solver_name = ["Ghost full", "Direct", "Ghost half"]


# In[3]:


rt = [{}, {}]


# In[4]:


for j in range(len(mu_name)):
    for theta in theta_list:

        rt[0][("full", j, theta)] = []
        rt[1][("full", j, theta)] = []

        for n in n_list[j]:

            norm_2_mask = numpy.ones(n+1)
            norm_2_mask[[0, -1]] /= numpy.sqrt(2)
            
            h = w / n

            if j == 0:
                m = d * n / 4.0
            elif j == 1:
                m = 15.0 * d * n**2
                
            m = int(m - 1.0e-5) + 1
            p =  (n * m - 1) // 150000000 + 1
            tau = d / m

            x = numpy.linspace(0.0, w, n+1)[:, None]
            ldu = m // p + 2
            u = numpy.zeros((n+1, ldu))
            u[:, 0] = models.calc_ana_sol_3(x, 0.0)[:, 0]

            print("j = {}, theta = {}, m = {}, n = {} started, {} chunks".format(j, theta, m, n, p))

            for k in range(p):

                m_l, m_u = (k * m) // p, ((k+1) * m) // p
                t = numpy.linspace(tau * m_l, tau * m_u, m_u - m_l + 1)[None, :]

                a = models.calc_coef_3(x, t)
                f = models.calc_sour_3(x, t)
                alpha = models.calc_alpha_3(t)
                g = models.calc_grad_3(t)

                exts.para_theta_ghost_full_wrapper(n, m_u - m_l, w, tau * (m_u - m_l), a, f, alpha, g, theta, u, ldu)

                u[:, 0] = u[:, m_u - m_l]

            u_m_sol = u[:, 0]
            u_m_ana = models.calc_ana_sol_3(x, d)[:, 0]
            rt[0][("full", j, theta)].append(numpy.linalg.norm(u_m_sol - u_m_ana, numpy.infty))
            rt[1][("full", j, theta)].append(numpy.linalg.norm((u_m_sol - u_m_ana) * norm_2_mask, 2.0) * numpy.sqrt(h),)

            del(u_m_sol)
            del(u_m_ana)
            del(u)
            del(f)
            del(a)


# In[5]:


for j in range(len(mu_name)):
    for theta in theta_list:

        rt[0][("direct", j, theta)] = []
        rt[1][("direct", j, theta)] = []

        for n in n_list[j]:

            norm_2_mask = numpy.ones(n+1)
            norm_2_mask[[0, -1]] /= numpy.sqrt(2)
            
            h = w / n

            if j == 0:
                m = d * n
            elif j == 1:
                m = 12.0 * d * n**2
                
            m = int(m - 1.0e-5) + 1
            p =  (n * m - 1) // 150000000 + 1
            tau = d / m

            x = numpy.linspace(0.0, w, n+1)[:, None]
            ldu = m // p + 2
            u = numpy.zeros((n+1, ldu))
            u[:, 0] = models.calc_ana_sol_3(x, 0.0)[:, 0]

            print("j = {}, theta = {}, m = {}, n = {} started, {} chunks".format(j, theta, m, n, p))

            for k in range(p):

                m_l, m_u = (k * m) // p, ((k+1) * m) // p
                t = numpy.linspace(tau * m_l, tau * m_u, m_u - m_l + 1)[None, :]

                a = models.calc_coef_3(x, t)
                f = models.calc_sour_3(x, t)
                alpha = models.calc_alpha_3(t)
                g = models.calc_grad_3(t)

                exts.para_theta_direct_wrapper(n, m_u - m_l, w, tau * (m_u - m_l), a, f, alpha, g, theta, u, ldu)

                u[:, 0] = u[:, m_u - m_l]

            u_m_sol = u[:, 0]
            u_m_ana = models.calc_ana_sol_3(x, d)[:, 0]
            rt[0][("direct", j, theta)].append(numpy.linalg.norm(u_m_sol - u_m_ana, numpy.infty))
            rt[1][("direct", j, theta)].append(numpy.linalg.norm((u_m_sol - u_m_ana) * norm_2_mask, 2.0) * numpy.sqrt(h),)

            del(u_m_sol)
            del(u_m_ana)
            del(u)
            del(f)
            del(a)


# In[6]:


for j in range(len(mu_name)):
    for theta in theta_list:

        rt[0][("half", j, theta)] = []
        rt[1][("half", j, theta)] = []

        for n in n_list[j]:

            h = w / n

            if j == 0:
                m = d * n
            elif j == 1:
                m = 12.0 * d * n**2
                
            m = int(m - 1.0e-5) + 1
            p =  (n * m - 1) // 150000000 + 1
            tau = d / m

            x = numpy.linspace(w / 2.0 / n, w - w / 2.0 / n, n)[:, None]
            ldu = m // p + 2
            u = numpy.zeros((n, ldu))
            u[:, 0] = models.calc_ana_sol_3(x, 0.0)[:, 0]

            print("j = {}, theta = {}, m = {}, n = {} started, {} chunks".format(j, theta, m, n, p))

            for k in range(p):

                m_l, m_u = (k * m) // p, ((k+1) * m) // p
                t = numpy.linspace(tau * m_l, tau * m_u, m_u - m_l + 1)[None, :]

                a = models.calc_coef_3(x, t)
                f = models.calc_sour_3(x, t)
                alpha = models.calc_alpha_3(t)
                g = models.calc_grad_3(t)

                exts.para_theta_ghost_half_wrapper(n, m_u - m_l, w, tau * (m_u - m_l), a, f, alpha, g, theta, u, ldu)

                u[:, 0] = u[:, m_u - m_l]

            u_m_sol = u[:, 0]
            u_m_ana = models.calc_ana_sol_3(x, d)[:, 0]
            rt[0][("half", j, theta)].append(numpy.linalg.norm(u_m_sol - u_m_ana, numpy.infty))
            rt[1][("half", j, theta)].append(numpy.linalg.norm(u_m_sol - u_m_ana, 2.0) * numpy.sqrt(h))

            del(u_m_sol)
            del(u_m_ana)
            del(u)
            del(f)
            del(a)


# In[7]:


with shelve.open("Result") as db:
    db[str((3, 2, "mu", "name"))] = mu_name
    db[str((3, 2, "n"))] = n_list
    db[str((3, 2, "theta"))] = theta_list
    db[str((3, 2, "theta", "name"))] = theta_name
    db[str((3, 2, "solver"))] = solver_list
    db[str((3, 2, "solver", "name"))] = solver_name
    db[str((3, 2, "normi"))] = rt[0]
    db[str((3, 2, "norm2"))] = rt[1]

