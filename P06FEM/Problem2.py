
# coding: utf-8

# In[1]:


import time
import numpy
from matplotlib import pyplot
import models
import solvers


# In[2]:


def calc_log_line(start, end, intc, order):
    return [start, end], [intc, intc * (end / start)**order]


# In[3]:


beta = models.get_beta_1()
cond = models.get_cond_1()


# In[4]:


ns = [2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024, 1448, 2048]


# In[5]:


es = [[], [], [], [], [], []]
ts = [[], []]
cs = [[], []]
titles = [
    "Triangle $L^{\infty}$", "Triangle $L^2$", "Triangle $H^1$",
    "Rectangle $L^{\infty}$", "Rectangle $L^2$", "Rectangle $H^1$",
]


# In[6]:


for n in ns:
    start = time.time()
    u_sol, ctr = solvers.driver_fem_tri(n, beta, cond)
    end = time.time()
    u_ana_all = models.driver_ana2_all_1(n)
    e = solvers.calc_err_tri(u_sol, *u_ana_all)
    es[0].append(e[0])
    es[1].append(e[1])
    es[2].append(e[2])
    ts[0].append(end - start)
    cs[0].append(ctr)
    print("n = {} finished".format(n))


# In[7]:


with open("Table1.tbl", "w") as f:
    for i, n in enumerate(ns):
        f.write("{} & {:.5e} & {:.5e} & {:.5e} & {:.3f} & {} \\\\\n\\hline\n".format(n, es[0][i], es[1][i], es[2][i], ts[0][i], cs[0][i]))


# In[8]:


for n in ns:
    start = time.time()
    u_sol, ctr = solvers.driver_fem_rect(n, beta, cond)
    end = time.time()
    u_ana_all = models.driver_ana2_all_1(n)
    e = solvers.calc_err_rect(u_sol, *u_ana_all)
    es[3].append(e[0])
    es[4].append(e[1])
    es[5].append(e[2])
    ts[1].append(end - start)
    cs[1].append(ctr)
    print("n = {} finished".format(n))


# In[9]:


with open("Table2.tbl", "w") as f:
    for i, n in enumerate(ns):
        f.write("{} & {:.5e} & {:.5e} & {:.5e} & {:.3f} & {} \\\\\n\\hline\n".format(n, es[3][i], es[4][i], es[5][i], ts[1][i], cs[1][i]))


# In[10]:


pyplot.figure(figsize=(6.0, 4.0))
for i in range(6):
    pyplot.plot(ns, es[i], label=titles[i])
    pyplot.scatter(ns, es[i], s=2.0)
pyplot.plot(*calc_log_line(ns[0], ns[-1], 1.0e0, -1.0), linewidth=0.5, color="black", label="Slope $-1$")
pyplot.plot(*calc_log_line(ns[0], ns[-1], 8.0e-2, -2.0), linewidth=0.5, color="black", linestyle="--", label="Slope $-2$")
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("$x$")
pyplot.ylabel("Error")
pyplot.xlim(left=1.0)
pyplot.legend(loc="lower left")
pyplot.savefig("Figure3.pgf")
pyplot.show()
pyplot.close()

