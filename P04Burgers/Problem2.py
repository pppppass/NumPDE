
# coding: utf-8

# In[1]:


import numpy
from matplotlib import pyplot
import models
import solvers


# In[2]:


ns = [10, 100, 1000]
ts = [1.0, 2.0, 3.0, 4.0, 5.0]


# In[3]:


for i, n in enumerate(ns):
    h = 1.0 / n
    tau = 0.1 / n
    x, u0 = models.get_sin_ana(n, 0.0, 3.0)
    pyplot.figure(figsize=(8.0, 12.0))
    ctr = 0
    us = [u0.copy() for j in range(4)]
    for j, t in enumerate(ts):
        ctr_next = int(t / tau + 1.0e-5)
        pyplot.subplot(3, 2, j+2)
        pyplot.title("$ t = {:.1f} $".format(t))
        u = us[0]
        for k in range(ctr_next - ctr):
            solvers.iter_rm(u, h, tau)
        pyplot.plot(x, u)
        print("t = {}, Richtmyer finished".format(t))
        u = us[1]
        for k in range(ctr_next - ctr):
            solvers.iter_lw(u, h, tau)
        pyplot.plot(x, u)
        print("t = {}, Lax-Wendroff finished".format(t))
        u = us[2]
        for k in range(ctr_next - ctr):
            solvers.iter_upwind_con(u, h, tau)
        pyplot.plot(x, u)
        print("t = {}, conservative upwind finished".format(t))
        u = us[3]
        for k in range(ctr_next - ctr):
            solvers.iter_upwind_non(u, h, tau)
        pyplot.plot(x, u)
        print("t = {}, non-conservative upwind finished".format(t))
        _, u = models.get_sin_ana(n, t, 3.0)
        pyplot.plot(x, u, color="black", linewidth=0.5, label="Analytical")
        pyplot.xlabel("$x$")
        pyplot.ylabel("$U$")
        ctr = ctr_next
    pyplot.subplot(3, 2, 1)
    pyplot.title("$ t = 0.0 $")
    pyplot.plot([], label="Richtmyer")
    pyplot.plot([], label="Lax--Wendroff")
    pyplot.plot([], label="Conservative upwind")
    pyplot.plot([], label="Non-conservative upwind")
    pyplot.plot(x, u0, color="black", linewidth=0.5, label="Analytical")
    pyplot.xlabel("$x$")
    pyplot.ylabel("$U$")
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.savefig("Figure{}.pgf".format(i+5))
    pyplot.show()
    pyplot.close()

