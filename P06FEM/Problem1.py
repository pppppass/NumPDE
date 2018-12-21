
# coding: utf-8

# In[1]:


import numpy
from matplotlib import pyplot
import models
import solvers


# In[2]:


beta = models.get_beta_1()
cond = models.get_cond_1()


# In[3]:


pyplot.figure(figsize=(8.0, 6.0))
u_sol, _ = solvers.driver_fem_tri(256, beta, cond)
u_sol2 = solvers.calc_int_tri(u_sol)
u_ana = models.driver_ana_1(256)
u_ana2, _, _ = models.driver_ana2_all_1(256)
pyplot.subplot(2, 2, 1)
pyplot.imshow(u_sol.transpose(), origin="lower")
pyplot.colorbar()
pyplot.xlabel("$x$")
pyplot.ylabel("$y$")
pyplot.subplot(2, 2, 2)
pyplot.imshow((u_sol - u_ana).transpose(), origin="lower")
pyplot.colorbar()
pyplot.xlabel("$x$")
pyplot.ylabel("$y$")
pyplot.tight_layout()
pyplot.subplot(2, 2, 3)
pyplot.imshow((u_sol2 - u_ana2).transpose(), origin="lower")
pyplot.colorbar()
pyplot.xlabel("$x$")
pyplot.ylabel("$y$")
u_sol, _ = solvers.driver_fem_tri(16, beta, cond)
u_sol2 = solvers.calc_int_tri(u_sol)
u_ana = models.driver_ana_1(16)
u_ana2, _, _ = models.driver_ana2_all_1(16)
pyplot.subplot(2, 2, 4)
pyplot.imshow((u_sol2 - u_ana2).transpose(), origin="lower")
pyplot.colorbar()
pyplot.xlabel("$x$")
pyplot.ylabel("$y$")
pyplot.tight_layout()
pyplot.savefig("Figure1.pgf")


# In[4]:


pyplot.figure(figsize=(8.0, 6.0))
u_sol, _ = solvers.driver_fem_rect(256, beta, cond)
u_sol2 = solvers.calc_int_rect(u_sol)
u_ana = models.driver_ana_1(256)
u_ana2, _, _ = models.driver_ana2_all_1(256)
pyplot.subplot(2, 2, 1)
pyplot.imshow(u_sol.transpose(), origin="lower")
pyplot.colorbar()
pyplot.xlabel("$x$")
pyplot.ylabel("$y$")
pyplot.subplot(2, 2, 2)
pyplot.imshow((u_sol - u_ana).transpose(), origin="lower")
pyplot.colorbar()
pyplot.xlabel("$x$")
pyplot.ylabel("$y$")
pyplot.tight_layout()
pyplot.subplot(2, 2, 3)
pyplot.imshow((u_sol2 - u_ana2).transpose(), origin="lower")
pyplot.colorbar()
pyplot.xlabel("$x$")
pyplot.ylabel("$y$")
u_sol, _ = solvers.driver_fem_rect(16, beta, cond)
u_sol2 = solvers.calc_int_rect(u_sol)
u_ana = models.driver_ana_1(16)
u_ana2, _, _ = models.driver_ana2_all_1(16)
pyplot.subplot(2, 2, 4)
pyplot.imshow((u_sol2 - u_ana2).transpose(), origin="lower")
pyplot.colorbar()
pyplot.xlabel("$x$")
pyplot.ylabel("$y$")
pyplot.tight_layout()
pyplot.savefig("Figure2.pgf")

