
# coding: utf-8

# In[1]:


import numpy
from matplotlib import pyplot
import models
import solvers


# In[2]:


def filter_array(array, upper):
    l = []
    for e in array:
        if numpy.abs(e) < upper:
            l.append(e)
        else:
            l.append(numpy.infty)
    return l


# In[3]:


n = 1000
h = 1.0 / n
tau = 0.1 / n
d = 10.0
m = 100
ts = numpy.linspace(0.0, d, m+1)


# In[4]:


ss = [solvers.iter_rm, solvers.iter_lw, solvers.iter_upwind_con, solvers.iter_upwind_non]
titles = ["Richtmyer", "Lax--Wendroff", "Conservative upwind", "Non-conservative upwind"]


# In[5]:


e_2s, e_infs = [], []


# In[6]:


for i, s in enumerate(ss):
    x, u = models.get_sin_ana(n, 0.0)
    e_2s.append([])
    e_infs.append([])
    ctr = 0
    for j, t in enumerate(ts):
        ctr_next = int(t / tau + 1.0e-5)
        for k in range(ctr_next - ctr):
            s(u, h, tau)
        _, u_std = models.get_sin_ana(n, t)
        e_2s[-1].append(numpy.sqrt(h) * numpy.linalg.norm(u - u_std))
        e_infs[-1].append(numpy.linalg.norm(u - u_std, numpy.infty))
        ctr = ctr_next
        print("t = {} finished".format(t))
    print("{} finished".format(titles[i]))


# In[9]:


pyplot.figure(figsize=(6.0, 4.0))
for i in range(4):
    pyplot.plot(ts, filter_array(e_2s[i], 0.1), label=titles[i])
    pyplot.scatter(ts, filter_array(e_2s[i], 0.1), s=2.0)
pyplot.legend()
pyplot.xlabel("$t$")
pyplot.ylabel("Error")
pyplot.savefig("Figure8.pgf")
pyplot.show()
pyplot.close()


# In[12]:


pyplot.figure(figsize=(6.0, 4.0))
for i in range(4):
    pyplot.plot(ts, filter_array(e_infs[i], 10.0), label=titles[i])
    pyplot.scatter(ts, filter_array(e_infs[i], 10.0), s=2.0)
pyplot.legend()
pyplot.xlabel("$t$")
pyplot.ylabel("Error")
pyplot.savefig("Figure9.pgf")
pyplot.show()
pyplot.close()

