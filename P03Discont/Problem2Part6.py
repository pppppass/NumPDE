
# coding: utf-8

# In[1]:


import shelve
import numpy
import models
import exts


# In[2]:


d_list = [0.1, 1.0, 10.0]
d_name = ["0.1", "1.0", "10.0"]
mu_name = [" \\tau / h = 1 / 5 ", " \\tau / h^2 = 1 / 5"]
n_list = [4, 5, 8, 11, 16, 22, 32, 45, 64, 90, 128, 181, 256, 362, 512]
theta_name = ["0", " 1 / 2 ", " 1 / 2 - 1 / 12 \\mu ", "1"]


# In[3]:


rt = [{}, {}, {}]


# In[4]:


for d in d_list:
    for j in range(2):
        for i in range(4):
            rt[0][str((d, j, i))] = []
            rt[1][str((d, j, i))] = []
        rt[2][str((d, j))] = []


# In[5]:


for n in n_list:
    h = 1.0 / n
    for d in d_list:
        for j in range(2):
            
            if j == 0:
                m = 5 * n
            elif j == 1:
                m = 5 * n**2

            u = models.calc_init_3(n)
            exts.para_theta_model_wrapper(n, int(m * d + 0.5), d, 0.0, u)
            u_ana = models.calc_approx_3(n, d)
            rt[0][str((d, j, 0))].append(numpy.linalg.norm(u - u_ana, numpy.infty))
            rt[1][str((d, j, 0))].append(numpy.linalg.norm(u - u_ana, 2.0) * numpy.sqrt(h))

            u = models.calc_init_3(n)
            exts.para_theta_model_wrapper(n, int(m * d + 0.5), d, 0.5, u)
            u_ana = models.calc_approx_3(n, d)
            rt[0][str((d, j, 1))].append(numpy.linalg.norm(u - u_ana, numpy.infty))
            rt[1][str((d, j, 1))].append(numpy.linalg.norm(u - u_ana, 2.0) * numpy.sqrt(h))

            tau = 1.0 / m
            mu = tau / h**2
            theta = 1.0 / 2.0 - 1.0 / 12.0 / mu
            if theta >= 0.0:
                u = models.calc_init_3(n)
                exts.para_theta_model_wrapper(n, int(m * d + 0.5), d, theta, u)
                u_ana = models.calc_approx_3(n, d)
                rt[0][str((d, j, 2))].append(numpy.linalg.norm(u - u_ana, numpy.infty))
                rt[1][str((d, j, 2))].append(numpy.linalg.norm(u - u_ana, 2.0) * numpy.sqrt(h))
            else:
                rt[0][str((d, j, 2))].append(numpy.infty)
                rt[1][str((d, j, 2))].append(numpy.infty)

            u = models.calc_init_3(n)
            exts.para_theta_model_wrapper(n, int(m * d + 0.5), d, 1.0, u)
            u_ana = models.calc_approx_3(n, d)
            rt[0][str((d, j, 3))].append(numpy.linalg.norm(u - u_ana, numpy.infty))
            rt[1][str((d, j, 3))].append(numpy.linalg.norm(u - u_ana, 2.0) * numpy.sqrt(h))

            rt[2][str((d, j))].append(int(m * d + 0.5) * n)
            
            print("d = {}, n = {}, j = {} finished".format(d, n, j))


# In[7]:


with shelve.open("Result") as db:
    db[str((2, 6, "d"))] = d_list
    db[str((2, 6, "d", "name"))] = d_name
    db[str((2, 6, "mu", "name"))] = mu_name
    db[str((2, 6, "n"))] = n_list
    db[str((2, 6, "theta", "name"))] = theta_name
    db[str((2, 6, "normi"))] = rt[0]
    db[str((2, 6, "norm2"))] = rt[1]
    db[str((2, 6, "comp"))] = rt[2]

