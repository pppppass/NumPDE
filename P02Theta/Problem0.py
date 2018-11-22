
# coding: utf-8

# In[1]:


import numpy
import models
import shelve


# In[2]:


w, d = 1.0, 2.0
n, m = 100, 200
h, tau = w / n, d / m


# In[3]:


x = numpy.linspace(0.0, w, n+1)[:, None]
t = numpy.linspace(0.0, d, m+1)[None, :]


# In[4]:


u_ana = models.calc_ana_sol_3(x, t)
numpy.save("Result01.npy", u_ana)


# In[5]:


t_name = [0, 1, 10, 100, 1000]
t = numpy.array([0.0, 1.0, 10.0, 100.0, 1000.0])


# In[6]:


u_t = models.calc_ana_sol_3(x, t)
numpy.save("Result02.npy", u_t)


# In[7]:


with shelve.open("Result") as db:
    db[str((0, "t", "name"))] = t_name

