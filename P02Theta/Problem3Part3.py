
# coding: utf-8

# In[1]:


import numpy
import models
import exts


# In[2]:


theta = 1.0 / 2.0


# In[3]:


w, d = 1.0, 2.0
n, m = 100, 200
h, tau = w / n, d / m


# In[4]:


x = numpy.linspace(0.0, w, n+1)[:, None]
t = numpy.linspace(0.0, d, m+1)[None, :]

u_ana = models.calc_ana_sol_3(x, t)

u = numpy.zeros((n+1, m+1))
u[:, 0] = models.calc_ana_sol_3(x, 0.0)[:, 0]
            
a = models.calc_coef_3(x, t)
f = models.calc_sour_3(x, t)
alpha = models.calc_alpha_3(t)
g = models.calc_grad_3(t)

exts.para_theta_direct_wrapper(n, m, w, d, a, f, alpha, g, theta, u, m+1)


# In[7]:


numpy.save("Result07.npy", u_ana)
numpy.save("Result08.npy", u)


# In[8]:


x = numpy.linspace(w / 2.0 / n, w - w / 2.0 / n, n)[:, None]
t = numpy.linspace(0.0, d, m+1)[None, :]

u_ana = models.calc_ana_sol_3(x, t)

u = numpy.zeros((n, m+1))
u[:, 0] = models.calc_ana_sol_3(x, 0.0)[:, 0]
            
a = models.calc_coef_3(x, t)
f = models.calc_sour_3(x, t)
alpha = models.calc_alpha_3(t)
g = models.calc_grad_3(t)

exts.para_theta_ghost_half_wrapper(n, m, w, d, a, f, alpha, g, theta, u, m+1)


# In[9]:


numpy.save("Result09.npy", u_ana)
numpy.save("Result10.npy", u)

