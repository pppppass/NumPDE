
# coding: utf-8

# In[1]:


import shelve
import numpy
import matplotlib
matplotlib.use("pgf")
from matplotlib import pyplot


# In[2]:


def filter_array(array, lamda):
    l = []
    for e in array:
        if lamda(e):
            l.append(e)
        else:
            l.append(numpy.infty)
    return l


# In[3]:


def calc_log_line(start, end, intc, order):
    return [start, end], [intc, intc * (end / start)**order]


# In[4]:


with shelve.open("Result") as db:
    t_name = db[str((0, "t", "name"))]
u_ana = numpy.load("Result01.npy")
u_t = numpy.load("Result02.npy")


# In[5]:


pyplot.figure(figsize=(6.0, 3.0))
pyplot.imshow(u_ana, extent=(0.0, 2.0, 0.0, 1.0))
pyplot.ylabel("$x$")
pyplot.xlabel("$t$")
pyplot.colorbar()
pyplot.savefig("Figure01.pgf")
pyplot.show()


# In[6]:


pyplot.figure(figsize=(6.0, 4.0))
for i, t in enumerate(t_name):
    pyplot.plot(numpy.linspace(0.0, 1.0, 101), u_t[:, i], label="$ t = {} $".format(t))
pyplot.xlabel("$x$")
pyplot.ylabel("$u$")
pyplot.legend()
pyplot.savefig("Figure02.pgf")
pyplot.show()


# In[7]:


with shelve.open("Result") as db:
    d = db[str((1, 1, "d"))]
    theta_list = db[str((1, 1, "theta"))]
    mu_list = db[str((1, 1, "mu"))]
    t_list = db[str((1, 1, "t"))]
    t_name = db[str((1, 1, "t", "name"))]
    m_list = db[str((1, 1, "m"))]
    normi_list = db[str((1, 1, "normi"))]
    norm2_list = db[str((1, 1, "norm2"))]


# In[8]:


for i, theta in enumerate(theta_list):
    with open("Table1{}.tbl".format(i+1), "w") as f:
        f.write("$\\mu$ & " + " & ".join(["$ t = {} $".format(t) for t in t_name]))
        f.write("\\\\\n")
        f.write("\\hline\n")
        for m in m_list[theta]:
            f.write("{:.5f} & ".format((d / m) / (1.0 / 100.0)**2))
            f.write(" & ".join(["{:.5e}".format(normi_list[(theta, m, t)]) for t in t_list]))
            f.write("\\\\\n")
            f.write("\\hline\n")


# In[9]:


for i, theta in enumerate(theta_list):
    with open("Table2{}.tbl".format(i+1), "w") as f:
        f.write("$\\mu$ & " + " & ".join(["$ t = {} $".format(t) for t in t_name]))
        f.write("\\\\\n")
        f.write("\\hline\n")
        for m in m_list[theta]:
            f.write("{:.5f} & ".format((d / m) / (1.0 / 100.0)**2))
            f.write(" & ".join(["{:.5e}".format(norm2_list[(theta, m, t)]) for t in t_list]))
            f.write("\\\\\n")
            f.write("\\hline\n")


# In[10]:


with shelve.open("Result") as db:
    mu_list = db[str((1, 2, "mu"))]
    p_name = db[str((1, 2, "p", "name"))]
    norm2_list = db[str((1, 2, "norm2"))]
    t_m_list = db[str((1, 2, "t_m"))]


# In[11]:


pyplot.figure(figsize=(7.0, 4.0))
pyplot.subplots_adjust(right=0.7)
for i, mu in enumerate(mu_list):
    pyplot.plot(t_m_list[mu], norm2_list[mu], label="$ p = {} $".format(p_name[i]))
pyplot.semilogy()
pyplot.ylim(1.0e-7, 1.0e-1)
pyplot.xlabel("$t$")
pyplot.ylabel("<LABEL1~~~~>")
pyplot.legend(loc="center right", bbox_to_anchor=(1.30, 0.5))
pyplot.savefig("Figure03.pgf")
pyplot.show()


# In[12]:


with shelve.open("Result") as db:
    t_list = db[str((2, 1, "t"))]
    t_name = db[str((2, 1, "t", "name"))]
    theta_list = db[str((2, 1, "theta"))]
    theta_name = db[str((2, 1, "theta", "name"))]
    m_list = db[str((2, 1, "m"))]
    normi_list = db[str((2, 1, "normi"))]
    norm2_list = db[str((2, 1, "norm2"))]


# In[13]:


pyplot.figure(figsize=(8.0, 8.0))
for j, t in enumerate(t_list):
    pyplot.subplot(2, 2, j+1)
    pyplot.title("$ t = {} $".format(t_name[j]))
    for i, theta in enumerate(theta_list):
        pyplot.plot(m_list, filter_array(normi_list[(t, theta)], lambda x: x < 1.0e-1), label="$ \\theta = {} $".format(theta_name[i]))
        pyplot.scatter(m_list, filter_array(normi_list[(t, theta)], lambda x: x < 1.0e-1), s=2.0)
    pyplot.plot([], color="black", linewidth=0.5, label="Slope $-1$")
    pyplot.plot([], color="black", linewidth=0.5, linestyle="--", label="Slope $-2$")
    if j == 0:
        pyplot.plot(*calc_log_line(1.0e3, 3.0e6, 2.0e-1, -1), color="black", linewidth=0.5)
        pyplot.plot(*calc_log_line(1.0e3, 3.0e4, 1.0e-1, -2), color="black", linewidth=0.5, linestyle="--")
    elif j == 1:
        pyplot.plot(*calc_log_line(1.0e3, 3.0e5, 2.0e-2, -1), color="black", linewidth=0.5)
        pyplot.plot(*calc_log_line(1.0e3, 3.0e3, 5.0e-3, -2), color="black", linewidth=0.5, linestyle="--")
    elif j == 2:
        pyplot.plot(*calc_log_line(1.0e3, 1.0e4, 1.0e-3, -1), color="black", linewidth=0.5)
    elif j == 3:
        pyplot.plot(*calc_log_line(1.0e3, 3.0e3, 3.0e-5, -1), color="black", linewidth=0.5)
    pyplot.semilogy()
    pyplot.semilogx()
    pyplot.xlabel("$M$")
    pyplot.ylabel("<LABEL8~~~>")
pyplot.legend(loc="right")
pyplot.tight_layout()
pyplot.savefig("Figure17.pgf")
pyplot.show()


# In[14]:


pyplot.figure(figsize=(8.0, 8.0))
for j, t in enumerate(t_list):
    pyplot.subplot(2, 2, j+1)
    pyplot.title("$ t = {} $".format(t_name[j]))
    for i, theta in enumerate(theta_list):
        pyplot.plot(m_list, filter_array(norm2_list[(t, theta)], lambda x: x < 1.0e-1), label="$ \\theta = {} $".format(theta_name[i]))
        pyplot.scatter(m_list, filter_array(norm2_list[(t, theta)], lambda x: x < 1.0e-1), s=2.0)
    pyplot.plot([], color="black", linewidth=0.5, label="Slope $-1$")
    pyplot.plot([], color="black", linewidth=0.5, linestyle="--", label="Slope $-2$")
    if j == 0:
        pyplot.plot(*calc_log_line(1.0e3, 3.0e6, 2.0e-1, -1), color="black", linewidth=0.5)
        pyplot.plot(*calc_log_line(1.0e3, 1.0e5, 1.0e-1, -2), color="black", linewidth=0.5, linestyle="--")
    elif j == 1:
        pyplot.plot(*calc_log_line(1.0e3, 3.0e5, 2.0e-2, -1), color="black", linewidth=0.5)
        pyplot.plot(*calc_log_line(1.0e3, 1.0e4, 5.0e-3, -2), color="black", linewidth=0.5, linestyle="--")
    elif j == 2:
        pyplot.plot(*calc_log_line(1.0e3, 2.0e4, 1.0e-3, -1), color="black", linewidth=0.5)
    elif j == 3:
        pyplot.plot(*calc_log_line(1.0e3, 3.0e3, 3.0e-5, -1), color="black", linewidth=0.5)
    pyplot.semilogy()
    pyplot.semilogx()
    pyplot.xlabel("$M$")
    pyplot.ylabel("<LABEL2~~~>")
pyplot.legend(loc="right")
pyplot.tight_layout()
pyplot.savefig("Figure04.pgf")
pyplot.show()


# In[15]:


with shelve.open("Result") as db:
    r_list = db[str((2, 2, "r"))]
    r_name = db[str((2, 2, "r", "name"))]
    n_list = db[str((2, 2, "n"))]
    theta_list = db[str((2, 2, "theta"))]
    theta_name = db[str((2, 2, "theta", "name"))]
    mu_list = db[str((2, 2, "mu"))]
    mu_name = db[str((2, 2, "mu", "name"))]
    normi_list = db[str((2, 2, "normi"))]
    norm2_list = db[str((2, 2, "norm2"))]
    time_list = db[str((2, 2, "time"))]
    comp_list = db[str((2, 2, "complex"))]


# In[16]:


for k in range(2):
    pyplot.figure(figsize=(8.0, 8.0))
    for i, theta in enumerate(theta_list):
        if i // 4 != k:
            continue
        pyplot.subplot(2, 2, i % 4 + 1)
        pyplot.title("$ \\theta = {} $".format(theta_name[i]))
        for l, r in enumerate(r_list):
            for j, mu in enumerate(mu_list[l]):
                pyplot.plot(n_list[l], filter_array(normi_list[(theta, r, mu)], lambda x: x < 1.0e-1), label=("$" + r_name[l] + mu_name[l][j] + "$"))
                pyplot.scatter(n_list[l], filter_array(normi_list[(theta, r, mu)], lambda x: x < 1.0e-1), s=2.0)
        pyplot.plot([], color="black", linewidth=0.5, label="Slope $-1$")
        pyplot.plot([], color="black", linewidth=0.5, linestyle="--", label="Slope $-2$")
        if i in [4]:
            pyplot.plot(*calc_log_line(50.0, 12800.0, 1.0e-3, -1), color="black", linewidth=0.5)
        elif i in [5]:
            pyplot.plot(*calc_log_line(50.0, 12800.0, 1.8e-3, -1), color="black", linewidth=0.5)
        elif i in [6]:
            pyplot.plot(*calc_log_line(50.0, 12800.0, 3.0e-3, -1), color="black", linewidth=0.5)
        if i in [0, 1, 2, 4, 5, 6]:
            pyplot.plot(*calc_log_line(3.0, 400.0, 2.0e-2, -2), color="black", linestyle="--", linewidth=0.5)
        elif i in [3]:
            pyplot.plot(*calc_log_line(3.0, 12800.0, 1.0e-2, -2), color="black", linestyle="--", linewidth=0.5)
        pyplot.semilogx()
        pyplot.semilogy()
        pyplot.xlabel("$N$")
        pyplot.ylabel("<LABEL3~~~>")
    pyplot.tight_layout()
    if k == 1:
        pyplot.legend(loc="right", bbox_to_anchor=(2.0, 0.5))
    pyplot.savefig("Figure05{}.pgf".format(k + 1))
    pyplot.show()


# In[17]:


for k in range(2):
    pyplot.figure(figsize=(8.0, 8.0))
    for i, theta in enumerate(theta_list):
        if i // 4 != k:
            continue
        pyplot.subplot(2, 2, i % 4 + 1)
        pyplot.title("$ \\theta = {} $".format(theta_name[i]))
        for l, r in enumerate(r_list):
            for j, mu in enumerate(mu_list[l]):
                pyplot.plot(n_list[l], filter_array(norm2_list[(theta, r, mu)], lambda x: x < 1.0e-1), label=("$" + r_name[l] + mu_name[l][j] + "$"))
                pyplot.scatter(n_list[l], filter_array(norm2_list[(theta, r, mu)], lambda x: x < 1.0e-1), s=2.0)
        pyplot.plot([], color="black", linewidth=0.5, label="Slope $-1$")
        pyplot.plot([], color="black", linewidth=0.5, linestyle="--", label="Slope $-2$")
        if i in [4]:
            pyplot.plot(*calc_log_line(50.0, 12800.0, 1.0e-3, -1), color="black", linewidth=0.5)
        elif i in [5]:
            pyplot.plot(*calc_log_line(50.0, 12800.0, 1.8e-3, -1), color="black", linewidth=0.5)
        elif i in [6]:
            pyplot.plot(*calc_log_line(50.0, 12800.0, 3.0e-3, -1), color="black", linewidth=0.5)
        if i in [0, 1, 2, 4, 5, 6]:
            pyplot.plot(*calc_log_line(3.0, 400.0, 2.0e-2, -2), color="black", linestyle="--", linewidth=0.5)
        elif i in [3]:
            pyplot.plot(*calc_log_line(3.0, 12800.0, 1.0e-2, -2), color="black", linestyle="--", linewidth=0.5)
        pyplot.semilogx()
        pyplot.semilogy()
        pyplot.xlabel("$N$")
        pyplot.ylabel("<LABEL4~~~>")
    pyplot.tight_layout()
    if k == 1:
        pyplot.legend(loc="right", bbox_to_anchor=(2.0, 0.5))
    pyplot.savefig("Figure06{}.pgf".format(k + 1))
    pyplot.show()


# In[18]:


for k in range(2):
    pyplot.figure(figsize=(8.0, 8.0))
    for i, theta in enumerate(theta_list):
        if i // 4 != k:
            continue
        pyplot.subplot(2, 2, i % 4 + 1)
        pyplot.title("$ \\theta = {} $".format(theta_name[i]))
        for l, r in enumerate(r_list):
            for j, mu in enumerate(mu_list[l]):
                pyplot.plot(n_list[l], time_list[(theta, r, mu)], label=("$" + r_name[l] + mu_name[l][j] + "$"))
                pyplot.scatter(n_list[l], time_list[(theta, r, mu)], s=2.0)
        pyplot.plot([], color="black", linewidth=0.5, label="Slope $2$")
        pyplot.plot([], color="black", linewidth=0.5, linestyle="--", label="Slope $3$")
        pyplot.plot(*calc_log_line(100.0, 12800.0, 3.0e-4, 2), color="black", linewidth=0.5)
        pyplot.plot(*calc_log_line(10.0, 400.0, 1.5e-2, 3), color="black", linestyle="--", linewidth=0.5)
        pyplot.semilogx()
        pyplot.semilogy()
        pyplot.xlabel("$N$")
        pyplot.ylabel("<LABEL5~~~>")
    pyplot.tight_layout()
    if k == 1:
        pyplot.legend(loc="right", bbox_to_anchor=(2.0, 0.5))
    pyplot.savefig("Figure07{}.pgf".format(k + 1))
    pyplot.show()


# In[19]:


for k in range(2):
    pyplot.figure(figsize=(8.0, 8.0))
    for i, theta in enumerate(theta_list):
        if i // 4 != k:
            continue
        pyplot.subplot(2, 2, i % 4 + 1)
        pyplot.title("$ \\theta = {} $".format(theta_name[i]))
        for l, r in enumerate(r_list):
            for j, mu in enumerate(mu_list[l]):
                pyplot.plot(n_list[l], comp_list[(theta, r, mu)], label=("$" + r_name[l] + mu_name[l][j] + "$"))
                pyplot.scatter(n_list[l], comp_list[(theta, r, mu)], s=2.0)
        pyplot.plot([], color="black", linewidth=0.5, label="Slope $2$")
        pyplot.plot([], color="black", linewidth=0.5, linestyle="--", label="Slope $3$")
        pyplot.plot(*calc_log_line(3.0, 12800.0, 6.0, 2), color="black", linewidth=0.5)
        pyplot.plot(*calc_log_line(3.0, 400.0, 5.0e3, 3), color="black", linestyle="--", linewidth=0.5)
        pyplot.semilogx()
        pyplot.semilogy()
        pyplot.xlabel("$N$")
        pyplot.ylabel("Estimated complexity")
    pyplot.tight_layout()
    if k == 1:
        pyplot.legend(loc="right", bbox_to_anchor=(2.0, 0.5))
    pyplot.savefig("Figure08{}.pgf".format(k + 1))
    pyplot.show()


# In[ ]:


u_ana = numpy.load("Result03.npy")
u = numpy.load("Result04.npy")


# In[ ]:


pyplot.figure(figsize=(6.0, 3.0))
pyplot.imshow(u, extent=(0.0, 2.0, 0.0, 1.0))
pyplot.ylabel("$x$")
pyplot.xlabel("$t$")
pyplot.colorbar()
pyplot.savefig("Figure09.pgf")
pyplot.show()


# In[ ]:


pyplot.figure(figsize=(6.0, 3.0))
pyplot.imshow(u - u_ana, extent=(0.0, 2.0, 0.0, 1.0))
pyplot.ylabel("$x$")
pyplot.xlabel("$t$")
pyplot.colorbar()
pyplot.savefig("Figure10.pgf")
pyplot.show()


# In[ ]:


u_ana = numpy.load("Result05.npy")
u = numpy.load("Result06.npy")


# In[ ]:


pyplot.figure(figsize=(6.0, 3.0))
pyplot.imshow(u, extent=(0.0, 2.0, 0.0, 1.0))
pyplot.ylabel("$x$")
pyplot.xlabel("$t$")
pyplot.colorbar()
pyplot.savefig("Figure11.pgf")
pyplot.show()


# In[ ]:


pyplot.figure(figsize=(6.0, 3.0))
pyplot.imshow(u - u_ana, extent=(0.0, 2.0, 0.0, 1.0))
pyplot.ylabel("$x$")
pyplot.xlabel("$t$")
pyplot.colorbar()
pyplot.savefig("Figure12.pgf")
pyplot.show()


# In[ ]:


u_ana = numpy.load("Result07.npy")
u = numpy.load("Result08.npy")


# In[ ]:


pyplot.figure(figsize=(6.0, 3.0))
pyplot.imshow(u, extent=(0.0, 2.0, 0.0, 1.0))
pyplot.ylabel("$x$")
pyplot.xlabel("$t$")
pyplot.colorbar()
pyplot.savefig("Figure18.pgf")
pyplot.show()


# In[ ]:


pyplot.figure(figsize=(6.0, 3.0))
pyplot.imshow(u - u_ana, extent=(0.0, 2.0, 0.0, 1.0))
pyplot.ylabel("$x$")
pyplot.xlabel("$t$")
pyplot.colorbar()
pyplot.savefig("Figure19.pgf")
pyplot.show()


# In[ ]:


u_ana = numpy.load("Result09.npy")
u = numpy.load("Result10.npy")


# In[ ]:


pyplot.figure(figsize=(6.0, 3.0))
pyplot.imshow(u, extent=(0.0, 2.0, 0.0, 1.0))
pyplot.ylabel("$x$")
pyplot.xlabel("$t$")
pyplot.colorbar()
pyplot.savefig("Figure20.pgf")
pyplot.show()


# In[ ]:


pyplot.figure(figsize=(6.0, 3.0))
pyplot.imshow(u - u_ana, extent=(0.0, 2.0, 0.0, 1.0))
pyplot.ylabel("$x$")
pyplot.xlabel("$t$")
pyplot.colorbar()
pyplot.savefig("Figure21.pgf")
pyplot.show()


# In[ ]:


with shelve.open("Result") as db:
    t_list = db[str((3, 1, "t"))]
    t_name = db[str((3, 1, "t", "name"))]
    solver_list = db[str((3, 1, "solver"))]
    solver_name = db[str((3, 1, "solver", "name"))]
    m_list = db[str((3, 1, "m"))]
    normi_list = db[str((3, 1, "normi"))]
    norm2_list = db[str((3, 1, "norm2"))]


# In[ ]:


pyplot.figure(figsize=(8.0, 8.0))
for j, t in enumerate(t_list):
    pyplot.subplot(2, 2, j+1)
    pyplot.title("$ t = {} $".format(t_name[j]))
    for i, solver in enumerate(solver_list):
        pyplot.plot(m_list, filter_array(normi_list[(t, solver)], lambda x: x < 1.0e-1), label="{}".format(solver_name[i]))
        pyplot.scatter(m_list, filter_array(normi_list[(t, solver)], lambda x: x < 1.0e-1), s=2.0)
    pyplot.plot([], color="black", linewidth=0.5, label="Slope $-2$")
    if j in [0]:
        pyplot.plot(*calc_log_line(1000.0, 100000.0, 2.0e-2, -2), color="black", linewidth=0.5)
    elif j in [1]:
        pyplot.plot(*calc_log_line(1000.0, 10000.0, 1.5e-3, -2), color="black", linewidth=0.5)
    pyplot.semilogy()
    pyplot.semilogx()
    pyplot.xlabel("$M$")
    pyplot.ylabel("<LABEL7~~~>")
pyplot.legend()
pyplot.tight_layout()
pyplot.savefig("Figure16.pgf")
pyplot.show()


# In[ ]:


pyplot.figure(figsize=(8.0, 8.0))
for j, t in enumerate(t_list):
    pyplot.subplot(2, 2, j+1)
    pyplot.title("$ t = {} $".format(t_name[j]))
    for i, solver in enumerate(solver_list):
        pyplot.plot(m_list, filter_array(norm2_list[(t, solver)], lambda x: x < 1.0e-1), label="{}".format(solver_name[i]))
        pyplot.scatter(m_list, filter_array(norm2_list[(t, solver)], lambda x: x < 1.0e-1), s=2.0)
    pyplot.plot([], color="black", linewidth=0.5, label="Slope $-2$")
    if j in [0]:
        pyplot.plot(*calc_log_line(1000.0, 100000.0, 2.0e-2, -2), color="black", linewidth=0.5)
    elif j in [1]:
        pyplot.plot(*calc_log_line(1000.0, 10000.0, 1.5e-3, -2), color="black", linewidth=0.5)
    pyplot.semilogy()
    pyplot.semilogx()
    pyplot.xlabel("$M$")
    pyplot.ylabel("<LABEL6~~~>")
pyplot.legend()
pyplot.tight_layout()
pyplot.savefig("Figure13.pgf")
pyplot.show()


# In[ ]:


with shelve.open("Result") as db:
    mu_name = db[str((3, 2, "mu", "name"))]
    n_list = db[str((3, 2, "n"))]
    theta_list = db[str((3, 2, "theta"))]
    theta_name = db[str((3, 2, "theta", "name"))]
    solver_list = db[str((3, 2, "solver"))]
    solver_name = db[str((3, 2, "solver", "name"))]
    normi_list = db[str((3, 2, "normi"))]
    norm2_list = db[str((3, 2, "norm2"))]


# In[ ]:


pyplot.figure(figsize=(8.0, 8.0))
for i, solver in enumerate(solver_list):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("{}".format(solver_name[i]))
    for l, theta in enumerate(theta_list):
        for j, mu in enumerate(mu_name):
            pyplot.plot(n_list[j], filter_array(normi_list[(solver, j, theta)], lambda x: x < 1.0e-1), label=("$ \\theta = {}, {} $".format(theta_name[l], mu)))
            pyplot.scatter(n_list[j], filter_array(normi_list[(solver, j, theta)], lambda x: x < 1.0e-1), s=2.0)
    pyplot.plot([], color="black", linewidth=0.5, label="Slope $-2$", linestyle="--")
    pyplot.plot([], color="black", linewidth=0.5, label="Slope $-1$")
    if i in [0]:
        pyplot.plot(*calc_log_line(3.0, 25600.0, 1.0e-2, -2), color="black", linestyle="--", linewidth=0.5)
        pyplot.plot(*calc_log_line(50.0, 25600.0, 5.0e-4, -1), color="black", linewidth=0.5)
    elif i in [1]:
        pyplot.plot(*calc_log_line(12.0, 25600.0, 1.8e-2, -1), color="black", linewidth=0.5)
    elif i in [2]:
        pyplot.plot(*calc_log_line(3.0, 1600.0, 1.0e-2, -2), color="black", linestyle="--", linewidth=0.5)
        pyplot.plot(*calc_log_line(25.0, 25600.0, 8.0e-4, -1), color="black", linewidth=0.5)
    pyplot.semilogx()
    pyplot.semilogy()
pyplot.tight_layout()
pyplot.legend(loc="right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure14.pgf")
pyplot.show()


# In[ ]:


pyplot.figure(figsize=(8.0, 8.0))
for i, solver in enumerate(solver_list):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("{}".format(solver_name[i]))
    for l, theta in enumerate(theta_list):
        for j, mu in enumerate(mu_name):
            pyplot.plot(n_list[j], filter_array(norm2_list[(solver, j, theta)], lambda x: x < 1.0e-1), label=("$ \\theta = {}, {} $".format(theta_name[l], mu)))
            pyplot.scatter(n_list[j], filter_array(norm2_list[(solver, j, theta)], lambda x: x < 1.0e-1), s=2.0)
    pyplot.plot([], color="black", linewidth=0.5, label="Slope $-2$", linestyle="--")
    pyplot.plot([], color="black", linewidth=0.5, label="Slope $-1$")
    if i in [0]:
        pyplot.plot(*calc_log_line(3.0, 25600.0, 1.0e-2, -2), color="black", linestyle="--", linewidth=0.5)
        pyplot.plot(*calc_log_line(50.0, 25600.0, 5.0e-4, -1), color="black", linewidth=0.5)
    elif i in [1]:
        pyplot.plot(*calc_log_line(12.0, 25600.0, 1.8e-2, -1), color="black", linewidth=0.5)
    elif i in [2]:
        pyplot.plot(*calc_log_line(3.0, 1600.0, 1.0e-2, -2), color="black", linestyle="--", linewidth=0.5)
        pyplot.plot(*calc_log_line(25.0, 25600.0, 8.0e-4, -1), color="black", linewidth=0.5)
    pyplot.semilogx()
    pyplot.semilogy()
pyplot.tight_layout()
pyplot.legend(loc="right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure15.pgf")
pyplot.show()

