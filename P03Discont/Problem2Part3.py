
# coding: utf-8

# In[1]:


import shelve
import numpy
import models
import exts


# In[2]:


d_list = [0.01, 0.1, 1.0, 10.0]
d_name = ["0.01", "0.1", "1.0", "10.0"]
n = 100
h = 1.0 / n
m_list = [
    2, 3, 4, 6, 8, 12, 17, 25, 35, 50, 70, 100, 141, 200,
    282, 400, 565, 800, 1131, 1600, 2262, 3200, 4525, 6400,
    9050, 12800, 18101, 25600, 36203, 51200, 72407, 102400,
    144815, 204800, 289630, 409600, 579261, 819200
]
theta_name = ["0", " 1 / 2 ", " 1 / 2 - 1 / 12 \\mu ", "1"]


# In[3]:


rt = [{}, {}]


# In[4]:


for d in d_list:
    
    for i in range(4):
        rt[0][str((d, i))] = []    
        rt[1][str((d, i))] = []
    
    for m in m_list:
        
        u = models.calc_init_2(n)
        exts.para_theta_model_wrapper(n, int(m * d + 0.5), d, 0.0, u)
        u_ana = models.calc_approx_2(n, d)
        rt[0][str((d, 0))].append(numpy.linalg.norm(u - u_ana, numpy.infty))
        rt[1][str((d, 0))].append(numpy.linalg.norm(u - u_ana, 2.0) * numpy.sqrt(h))
        
        u = models.calc_init_2(n)
        exts.para_theta_model_wrapper(n, int(m * d + 0.5), d, 0.5, u)
        u_ana = models.calc_approx_2(n, d)
        rt[0][str((d, 1))].append(numpy.linalg.norm(u - u_ana, numpy.infty))
        rt[1][str((d, 1))].append(numpy.linalg.norm(u - u_ana, 2.0) * numpy.sqrt(h))
        
        tau = 1.0 / m
        mu = tau / h**2
        theta = 1.0 / 2.0 - 1.0 / 12.0 / mu
        if theta <= 1.0:
            u = models.calc_init_2(n)
            exts.para_theta_model_wrapper(n, int(m * d + 0.5), d, theta, u)
            u_ana = models.calc_approx_2(n, d)
            rt[0][str((d, 2))].append(numpy.linalg.norm(u - u_ana, numpy.infty))
            rt[1][str((d, 2))].append(numpy.linalg.norm(u - u_ana, 2.0) * numpy.sqrt(h))
        else:
            rt[0][str((d, 2))].append(numpy.infty)
            rt[1][str((d, 2))].append(numpy.infty)
        
        u = models.calc_init_2(n)
        exts.para_theta_model_wrapper(n, int(m * d + 0.5), d, 1.0, u)
        u_ana = models.calc_approx_2(n, d)
        rt[0][str((d, 3))].append(numpy.linalg.norm(u - u_ana, numpy.infty))
        rt[1][str((d, 3))].append(numpy.linalg.norm(u - u_ana, 2.0) * numpy.sqrt(h))
        
        print("d = {}, m = {} finished".format(d, m))


# In[5]:


with shelve.open("Result") as db:
    db[str((2, 3, "d"))] = d_list
    db[str((2, 3, "d", "name"))] = d_name
    db[str((2, 3, "m"))] = m_list
    db[str((2, 3, "theta", "name"))] = theta_name
    db[str((2, 3, "normi"))] = rt[0]
    db[str((2, 3, "norm2"))] = rt[1]

