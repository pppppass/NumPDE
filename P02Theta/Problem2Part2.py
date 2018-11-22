
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
import models
import exts


# In[2]:


w, d = 1.0, 10.0
r_list = [1.0, 2.0]
r_name = [" \\tau / h = ", " \\tau / h^2 = "]
n_list = [
    [3, 4, 6, 8, 12, 17, 25, 35, 50, 70, 100, 141, 200, 282, 400, 565, 800, 1131, 1600, 2262, 3200, 4525, 6400, 9050, 12800],
    [3, 4, 6, 8, 12, 17, 25, 35, 50, 70, 100, 141, 200, 282, 400],
]
theta_list = [0.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0, 3.0 / 4.0, 1.0]
theta_name = ["0", " 1 / 4 ", " 1 / 3 ", " 1 / 2 ", " 2 / 3 ", " 3 / 4 ", "1"]
mu_list = [
    [8.0, 4.0, 2.0, 1.0, 1.0 / 2.0],
    [2.0, 1.0, 1.0 / 2.0, 1.0 / 4.0, 1.0 / 8.0]
]
mu_name = [
    ["8", "4", "2", "1", " 1 / 2 "],
    ["2", "1", " 1 / 2 ", " 1 / 4 ", " 1 / 8"]
]


# In[3]:


rt = [{}, {}, {}, {}]


# In[4]:


for theta in theta_list:
    for l, r in enumerate(r_list):
        for mu_ in mu_list[l]:
            
            rt[0][(theta, r, mu_)] = []
            rt[1][(theta, r, mu_)] = []
            rt[2][(theta, r, mu_)] = []
            rt[3][(theta, r, mu_)] = []
            
            for n in n_list[l]:
                
                norm_2_mask = numpy.ones(n+1)
                norm_2_mask[[0, -1]] /= numpy.sqrt(2)
                h = w / n

                m = int(d / mu_ / h**r - 1.0e-5) + 1
                p =  (n * m - 1) // 150000000 + 1
                tau = d / m

                x = numpy.linspace(0.0, w, n+1)[:, None]
                ldu = m // p + 2
                u = numpy.zeros((n+1, ldu))
                u[:, 0] = models.calc_ana_sol_3(x, 0.0)[:, 0]

                print("theta = {}, m = {}, n = {} started, {} chunks".format(theta, m, n, p))
                start = time.time()

                for k in range(p):

                    m_l, m_u = (k * m) // p, ((k+1) * m) // p
                    t = numpy.linspace(tau * m_l, tau * m_u, m_u - m_l + 1)[None, :]

                    a = models.calc_coef_3(x, t)
                    f = models.calc_sour_3(x, t)
                    alpha = models.calc_alpha_3(t)
                    g = models.calc_grad_3(t)

                    exts.para_theta_ghost_full_wrapper(n, m_u - m_l, w, tau * (m_u - m_l), a, f, alpha, g, theta, u, ldu)

                    u[:, 0] = u[:, m_u - m_l]

                end = time.time()
                
                u_m_sol = u[:, 0]
                u_m_ana = models.calc_ana_sol_3(x, d)[:, 0]
                rt[0][(theta, r, mu_)].append(numpy.linalg.norm(u_m_sol - u_m_ana, numpy.infty))
                rt[1][(theta, r, mu_)].append(numpy.linalg.norm((u_m_sol - u_m_ana) * norm_2_mask, 2.0) * numpy.sqrt(h),)
                rt[2][(theta, r, mu_)].append(end - start)
                rt[3][(theta, r, mu_)].append(m * n)

                del(u_m_sol)
                del(u_m_ana)
                del(u)
                del(f)
                del(a)


# In[5]:


with shelve.open("Result") as db:
    db[str((2, 2, "r"))] = r_list
    db[str((2, 2, "r", "name"))] = r_name
    db[str((2, 2, "n"))] = n_list
    db[str((2, 2, "theta"))] = theta_list
    db[str((2, 2, "theta", "name"))] = theta_name
    db[str((2, 2, "mu"))] = mu_list
    db[str((2, 2, "mu", "name"))] = mu_name
    db[str((2, 2, "normi"))] = rt[0]
    db[str((2, 2, "norm2"))] = rt[1]
    db[str((2, 2, "time"))] = rt[2]
    db[str((2, 2, "complex"))] = rt[3]

