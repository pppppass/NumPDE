
# coding: utf-8

# In[1]:


import shelve
import numpy
import models


# In[2]:


n = 3000


# In[3]:


t_list = numpy.array([0.01, 0.1, 1, 10])
t_name = ["0", "0.01", "0.1", "1", "10"]


# In[4]:


u = numpy.concatenate([models.calc_init_1(n), models.calc_approx_1(n, t_list)], axis=1)
numpy.save("Result1.npy", u)


# In[5]:


u = numpy.concatenate([models.calc_init_2(n), models.calc_approx_2(n, t_list)], axis=1)
numpy.save("Result2.npy", u)


# In[6]:


u = numpy.concatenate([models.calc_init_3(n), models.calc_approx_3(n, t_list)], axis=1)
numpy.save("Result3.npy", u)


# In[7]:


with shelve.open("Result") as db:
    db[str((0, "n"))] = n
    db[str((0, "time", "name"))] = t_name

