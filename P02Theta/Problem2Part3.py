
# coding: utf-8

# In[7]:


import numpy
import models
import exts


# In[8]:


theta = 1.0 / 2.0


# In[9]:


w, d = 1.0, 2.0
n, m = 100, 200
h, tau = w / n, d / m


# In[10]:


x = numpy.linspace(0.0, w, n+1)[:, None]
t = numpy.linspace(0.0, d, m+1)[None, :]

u_ana = models.calc_ana_sol_3(x, t)

u = numpy.zeros((n+1, m+1))
u[:, 0] = models.calc_ana_sol_3(x, 0.0)[:, 0]
            
a = models.calc_coef_3(x, t)
f = models.calc_sour_3(x, t)
alpha = models.calc_alpha_3(t)
g = models.calc_grad_3(t)

exts.para_theta_ghost_full_wrapper(n, m, w, d, a, f, alpha, g, theta, u, m+1)


# In[11]:


numpy.save("Result03.npy", u_ana)
numpy.save("Result04.npy", u)


# In[12]:


w, d = 1.0, 2.0
n, m = 100, 160000
h, tau = w / n, d / m


# In[14]:


x = numpy.linspace(0.0, w, n+1)[:, None]
t = numpy.linspace(0.0, d, m+1)[None, :]

u_ana = models.calc_ana_sol_3(x, t)

u = numpy.zeros((n+1, m+1))
u[:, 0] = models.calc_ana_sol_3(x, 0.0)[:, 0]
            
a = models.calc_coef_3(x, t)
f = models.calc_sour_3(x, t)
alpha = models.calc_alpha_3(t)
g = models.calc_grad_3(t)

exts.para_theta_ghost_full_wrapper(n, m, w, d, a, f, alpha, g, theta, u, m+1)


# In[15]:


numpy.save("Result05.npy", u_ana[::100, :])
numpy.save("Result06.npy", u[::100, :])

