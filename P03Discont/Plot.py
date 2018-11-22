
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
    n = db[str((0, "n"))]
    t_name = db[str((0, "time", "name"))]
h = 1.0 / n
x = numpy.linspace(0.0, 1.0, n+1)


# In[5]:


u = numpy.load("Result1.npy")


# In[6]:


pyplot.figure(figsize=(6.0, 4.0))
for i, t in enumerate(t_name):
    pyplot.plot(x[1:-1], u[:, i], label="$ t = {} $".format(t))
pyplot.legend()
pyplot.xlabel("$x$")
pyplot.ylabel("$u$")
pyplot.savefig("Figure01.pgf")
pyplot.show()


# In[7]:


u = numpy.load("Result4.npy")


# In[8]:


pyplot.figure(figsize=(6.0, 4.0))
for i, t in enumerate(t_name):
    pyplot.plot(x[1:-1], u[:, i], label="$ t = {} $".format(t))
pyplot.legend()
pyplot.xlabel("$x$")
pyplot.ylabel("$u$")
pyplot.savefig("Figure22.pgf")
pyplot.show()


# In[9]:


with open("Table1.tbl", "w") as f:
    for i, t in enumerate(t_name):
        f.write("{} & {:.5e} & {:.5e} \\\\\n".format(
            t,
            numpy.linalg.norm(u[:, i], numpy.infty),
            numpy.linalg.norm(u[:, i], 2.0) * numpy.sqrt(h)
        ))
        f.write("\\hline\n")


# In[10]:


u = numpy.load("Result2.npy")


# In[11]:


pyplot.figure(figsize=(6.0, 4.0))
for i, t in enumerate(t_name):
    pyplot.plot(x[1:-1], u[:, i], label="$ t = {} $".format(t))
pyplot.legend()
pyplot.xlabel("$x$")
pyplot.ylabel("$u$")
pyplot.savefig("Figure02.pgf")
pyplot.show()


# In[12]:


with open("Table2.tbl", "w") as f:
    for i, t in enumerate(t_name):
        f.write("{} & {:.5e} & {:.5e} \\\\\n".format(
            t,
            numpy.linalg.norm(u[:, i], numpy.infty),
            numpy.linalg.norm(u[:, i], 2.0) * numpy.sqrt(h)
        ))
        f.write("\\hline\n")


# In[13]:


u = numpy.load("Result5.npy")


# In[14]:


pyplot.figure(figsize=(6.0, 4.0))
for i, t in enumerate(t_name):
    pyplot.plot(x[1:-1], u[:, i], label="$ t = {} $".format(t))
pyplot.legend()
pyplot.xlabel("$x$")
pyplot.ylabel("$u$")
pyplot.savefig("Figure23.pgf")
pyplot.show()


# In[15]:


u = numpy.load("Result3.npy")


# In[16]:


x = numpy.linspace(0.0, 1.0, n+1)
pyplot.figure(figsize=(6.0, 4.0))
for i, t in enumerate(t_name):
    pyplot.plot(x[1:-1], u[:, i], label="$ t = {} $".format(t))
pyplot.legend()
pyplot.xlabel("$x$")
pyplot.ylabel("$u$")
pyplot.savefig("Figure03.pgf")
pyplot.show()


# In[17]:


u = numpy.load("Result6.npy")


# In[18]:


pyplot.figure(figsize=(6.0, 4.0))
for i, t in enumerate(t_name):
    pyplot.plot(x[1:-1], u[:, i], label="$ t = {} $".format(t))
pyplot.legend()
pyplot.xlabel("$x$")
pyplot.ylabel("$u$")
pyplot.savefig("Figure24.pgf")
pyplot.show()


# In[19]:


with open("Table3.tbl", "w") as f:
    for i, t in enumerate(t_name):
        f.write("{} & {:.5e} & {:.5e} \\\\\n".format(
            t,
            numpy.linalg.norm(u[:, i], numpy.infty),
            numpy.linalg.norm(u[:, i], 2.0) * numpy.sqrt(h)
        ))
        f.write("\\hline\n")


# In[20]:


with shelve.open("Result") as db:
    d_list = db[str((2, 1, "d"))]
    d_name = db[str((2, 1, "d", "name"))]
    m_list = db[str((2, 1, "m"))]
    theta_name = db[str((2, 1, "theta", "name"))]
    normi_list = db[str((2, 1, "normi"))]
    norm2_list = db[str((2, 1, "norm2"))]


# In[21]:


pyplot.figure(figsize=(8.0, 8.0))
for i, d in enumerate(d_list):
    if i == 0:
        continue
    pyplot.subplot(2, 2, i)
    pyplot.title("$ L = {} $".format(d_name[i]))
    for j, theta in enumerate(theta_name):
        pyplot.plot(m_list, filter_array(normi_list[str((d, j))], lambda x: x < 1.0e-1), label="$ \\theta = {} $".format(theta))
        pyplot.scatter(m_list, filter_array(normi_list[str((d, j))], lambda x: x < 1.0e-1), s=2.0)
    if i == 1:
        pyplot.plot(*calc_log_line(75.0, 12800.0, 2.0e-3, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(50.0, 20000.0, 1.5e-4, -2.0), linewidth=0.5, color="black", linestyle="--")
    elif i == 2:
        pyplot.plot(*calc_log_line(75.0, 12800.0, 1.0e-6, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(2.0, 51200.0, 1.0e-3, -2.0), linewidth=0.5, color="black", linestyle="--")
    elif i == 3:
        pyplot.plot(*calc_log_line(2.0, 100.0, 1.0e-1, -2.0), linewidth=0.5, color="black", linestyle="--")
    pyplot.plot([], linewidth=0.5, color="black", label="Slope $-1$")
    pyplot.plot([], linewidth=0.5, color="black", linestyle="--", label="Slope $-2$")
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("$ M / L $")
    pyplot.ylabel("Error")
pyplot.tight_layout()
pyplot.legend(loc="center right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure04.pgf")
pyplot.show()


# In[22]:


pyplot.figure(figsize=(8.0, 8.0))
for i, d in enumerate(d_list):
    if i == 0:
        continue
    pyplot.subplot(2, 2, i)
    pyplot.title("$ L = {} $".format(d_name[i]))
    for j, theta in enumerate(theta_name):
        pyplot.plot(m_list, filter_array(norm2_list[str((d, j))], lambda x: x < 1.0e-1), label="$ \\theta = {} $".format(theta))
        pyplot.scatter(m_list, filter_array(norm2_list[str((d, j))], lambda x: x < 1.0e-1), s=2.0)
    if i == 1:
        pyplot.plot(*calc_log_line(75.0, 12800.0, 1.0e-3, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(50.0, 20000.0, 0.6e-4, -2.0), linewidth=0.5, color="black", linestyle="--")
    elif i == 2:
        pyplot.plot(*calc_log_line(75.0, 12800.0, 1.0e-6, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(2.0, 51200.0, 0.7e-3, -2.0), linewidth=0.5, color="black", linestyle="--")
    elif i == 3:
        pyplot.plot(*calc_log_line(2.0, 100.0, 0.3e-1, -2.0), linewidth=0.5, color="black", linestyle="--")
    pyplot.plot([], linewidth=0.5, color="black", label="Slope $-1$")
    pyplot.plot([], linewidth=0.5, color="black", linestyle="--", label="Slope $-2$")
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("$ M / L $")
    pyplot.ylabel("Error")
pyplot.tight_layout()
pyplot.legend(loc="center right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure05.pgf")
pyplot.show()


# In[23]:


with shelve.open("Result") as db:
    d_list = db[str((2, 2, "d"))]
    d_name = db[str((2, 2, "d", "name"))]
    mu_name = db[str((2, 2, "mu", "name"))]
    n_list = db[str((2, 2, "n"))]
    theta_name = db[str((2, 2, "theta", "name"))]
    normi_list = db[str((2, 2, "normi"))]
    norm2_list = db[str((2, 2, "norm2"))]
    comp_list = db[str((2, 2, "comp"))]


# In[24]:


pyplot.figure(figsize=(8.0, 8.0))
for i, d in enumerate(d_list):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ L = {} $".format(d_name[i]))
    for j, theta in enumerate(theta_name):
        for l, mu in enumerate(mu_name):
            pyplot.plot(n_list, filter_array(norm2_list[str((d, l, j))], lambda x: x < 1.0e-1), label="$ \\theta = {}, {} $".format(theta, mu))
            pyplot.scatter(n_list, filter_array(norm2_list[str((d, l, j))], lambda x: x < 1.0e-1), s=2.0)
    if i == 0:
        pyplot.plot(*calc_log_line(4.0, 512.0, 5.0e-3, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(4.0, 512.0, 1.2e-4, -2.0), linewidth=0.5, color="black", linestyle="--")
        pyplot.plot(*calc_log_line(4.0, 512.0, 1.2e-5, -4.0), linewidth=0.5, color="black", linestyle=":")
    elif i == 1:
        pyplot.plot(*calc_log_line(8.0, 512.0, 0.3e-5, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(4.0, 512.0, 2.0e-7, -2.0), linewidth=0.5, color="black", linestyle="--")
        pyplot.plot(*calc_log_line(4.0, 256.0, 3.0e-9, -4.0), linewidth=0.5, color="black", linestyle=":")
    elif i == 2:
        pyplot.plot(*calc_log_line(32.0, 512.0, 2.0e-44, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(4.0, 512.0, 3.0e-45, -2.0), linewidth=0.5, color="black", linestyle="--")
        pyplot.plot(*calc_log_line(4.0, 256.0, 1.0e-46, -4.0), linewidth=0.5, color="black", linestyle=":")
    pyplot.plot([], linewidth=0.5, color="black", label="Slope $-1$")
    pyplot.plot([], linewidth=0.5, color="black", linestyle="--", label="Slope $-2$")
    pyplot.plot([], linewidth=0.5, color="black", linestyle=":", label="Slope $-4$")
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlim(3.0, 768.0)
    pyplot.xlabel("$N$")
    pyplot.ylabel("Error")
pyplot.tight_layout()
pyplot.legend(loc="center right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure06.pgf")
pyplot.show()


# In[25]:


pyplot.figure(figsize=(8.0, 8.0))
for i, d in enumerate(d_list):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ L = {} $".format(d_name[i]))
    for j, theta in enumerate(theta_name):
        for l, mu in enumerate(mu_name):
            pyplot.plot(n_list, filter_array(norm2_list[str((d, l, j))], lambda x: x < 1.0e-1), label="$ \\theta = {}, {} $".format(theta, mu))
            pyplot.scatter(n_list, filter_array(norm2_list[str((d, l, j))], lambda x: x < 1.0e-1), s=2.0)
    if i == 0:
        pyplot.plot(*calc_log_line(4.0, 512.0, 5.0e-3, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(4.0, 512.0, 1.2e-4, -2.0), linewidth=0.5, color="black", linestyle="--")
        pyplot.plot(*calc_log_line(4.0, 512.0, 1.2e-5, -4.0), linewidth=0.5, color="black", linestyle=":")
    elif i == 1:
        pyplot.plot(*calc_log_line(8.0, 512.0, 0.3e-5, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(4.0, 512.0, 2.0e-7, -2.0), linewidth=0.5, color="black", linestyle="--")
        pyplot.plot(*calc_log_line(4.0, 256.0, 3.0e-9, -4.0), linewidth=0.5, color="black", linestyle=":")
    elif i == 2:
        pyplot.plot(*calc_log_line(32.0, 512.0, 2.0e-44, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(4.0, 512.0, 3.0e-45, -2.0), linewidth=0.5, color="black", linestyle="--")
        pyplot.plot(*calc_log_line(4.0, 256.0, 1.0e-46, -4.0), linewidth=0.5, color="black", linestyle=":")
    pyplot.plot([], linewidth=0.5, color="black", label="Slope $-1$")
    pyplot.plot([], linewidth=0.5, color="black", linestyle="--", label="Slope $-2$")
    pyplot.plot([], linewidth=0.5, color="black", linestyle=":", label="Slope $-4$")
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlim(3.0, 768.0)
    pyplot.xlabel("$N$")
    pyplot.ylabel("Error")
pyplot.tight_layout()
pyplot.legend(loc="center right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure07.pgf")
pyplot.show()


# In[26]:


pyplot.figure(figsize=(8.0, 8.0))
for i, d in enumerate(d_list):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ L = {} $".format(d_name[i]))
    for j, theta in enumerate(theta_name):
        for l, mu in enumerate(mu_name):
            pyplot.plot(comp_list[str((d, l))], filter_array(normi_list[str((d, l, j))], lambda x: x < 1.0e-1), label="$ \\theta = {}, {} $".format(theta, mu))
            pyplot.scatter(comp_list[str((d, l))], filter_array(normi_list[str((d, l, j))], lambda x: x < 1.0e-1), s=2.0)
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("Estimated complexity")
    pyplot.ylabel("Error")
pyplot.tight_layout()
pyplot.legend(loc="center right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure08.pgf")
pyplot.show()


# In[27]:


pyplot.figure(figsize=(8.0, 8.0))
for i, d in enumerate(d_list):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ L = {} $".format(d_name[i]))
    for j, theta in enumerate(theta_name):
        for l, mu in enumerate(mu_name):
            pyplot.plot(comp_list[str((d, l))], filter_array(norm2_list[str((d, l, j))], lambda x: x < 1.0e-1), label="$ \\theta = {}, {} $".format(theta, mu))
            pyplot.scatter(comp_list[str((d, l))], filter_array(norm2_list[str((d, l, j))], lambda x: x < 1.0e-1), s=2.0)
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("Estimated complexity")
    pyplot.ylabel("Error")
pyplot.tight_layout()
pyplot.legend(loc="center right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure09.pgf")
pyplot.show()


# In[28]:


with shelve.open("Result") as db:
    d_list = db[str((2, 3, "d"))]
    d_name = db[str((2, 3, "d", "name"))]
    m_list = db[str((2, 3, "m"))]
    theta_name = db[str((2, 3, "theta", "name"))]
    normi_list = db[str((2, 3, "normi"))]
    norm2_list = db[str((2, 3, "norm2"))]


# In[29]:


pyplot.figure(figsize=(8.0, 8.0))
for i, d in enumerate(d_list):
    if i == 0:
        continue
    pyplot.subplot(2, 2, i)
    pyplot.title("$ L = {} $".format(d_name[i]))
    for j, theta in enumerate(theta_name):
        pyplot.plot(m_list, filter_array(normi_list[str((d, j))], lambda x: x < 1.0e-1), label="$ \\theta = {} $".format(theta))
        pyplot.scatter(m_list, filter_array(normi_list[str((d, j))], lambda x: x < 1.0e-1), s=2.0)
    if i == 1:
        pyplot.plot(*calc_log_line(30.0, 12800.0, 4.0e-3, -1.0), linewidth=0.5, color="black")
    elif i == 2:
        pyplot.plot(*calc_log_line(75.0, 12800.0, 1.0e-6, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(2.0, 75.0, 1.0e-3, -2.0), linewidth=0.5, color="black", linestyle="--")
    pyplot.plot([], linewidth=0.5, color="black", label="Slope $-1$")
    pyplot.plot([], linewidth=0.5, color="black", linestyle="--", label="Slope $-2$")
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("$ M / L $")
    pyplot.ylabel("Error")
pyplot.tight_layout()
pyplot.legend(loc="center right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure10.pgf")
pyplot.show()


# In[30]:


pyplot.figure(figsize=(8.0, 8.0))
for i, d in enumerate(d_list):
    if i == 0:
        continue
    pyplot.subplot(2, 2, i)
    pyplot.title("$ L = {} $".format(d_name[i]))
    for j, theta in enumerate(theta_name):
        pyplot.plot(m_list, filter_array(norm2_list[str((d, j))], lambda x: x < 1.0e-1), label="$ \\theta = {} $".format(theta))
        pyplot.scatter(m_list, filter_array(norm2_list[str((d, j))], lambda x: x < 1.0e-1), s=2.0)
    if i == 1:
        pyplot.plot(*calc_log_line(30.0, 12800.0, 3.0e-3, -1.0), linewidth=0.5, color="black")
    elif i == 2:
        pyplot.plot(*calc_log_line(75.0, 12800.0, 1.5e-6, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(2.0, 75.0, 0.8e-3, -2.0), linewidth=0.5, color="black", linestyle="--")
    pyplot.plot([], linewidth=0.5, color="black", label="Slope $-1$")
    pyplot.plot([], linewidth=0.5, color="black", linestyle="--", label="Slope $-2$")
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("$ M / L $")
    pyplot.ylabel("Error")
pyplot.tight_layout()
pyplot.legend(loc="center right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure11.pgf")
pyplot.show()


# In[31]:


with shelve.open("Result") as db:
    d_list = db[str((2, 4, "d"))]
    d_name = db[str((2, 4, "d", "name"))]
    mu_name = db[str((2, 4, "mu", "name"))]
    n_list = db[str((2, 4, "n"))]
    theta_name = db[str((2, 4, "theta", "name"))]
    normi_list = db[str((2, 4, "normi"))]
    norm2_list = db[str((2, 4, "norm2"))]
    comp_list = db[str((2, 4, "comp"))]


# In[32]:


pyplot.figure(figsize=(8.0, 8.0))
for i, d in enumerate(d_list):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ L = {} $".format(d_name[i]))
    for j, theta in enumerate(theta_name):
        for l, mu in enumerate(mu_name):
            pyplot.plot(n_list, filter_array(normi_list[str((d, l, j))], lambda x: x < 1.0e-1), label="$ \\theta = {}, {} $".format(theta, mu))
            pyplot.scatter(n_list, filter_array(normi_list[str((d, l, j))], lambda x: x < 1.0e-1), s=2.0)
    if i == 0:
        pyplot.plot(*calc_log_line(4.0, 512.0, 5.0e-3, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(4.0, 512.0, 3.0e-4, -2.0), linewidth=0.5, color="black", linestyle="--")
    elif i == 1:
        pyplot.plot(*calc_log_line(16.0, 512.0, 0.15e-5, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(4.0, 512.0, 0.3e-7, -2.0), linewidth=0.5, color="black", linestyle="--")
    elif i == 2:
        pyplot.plot(*calc_log_line(64.0, 512.0, 1.0e-44, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(4.0, 512.0, 3.0e-45, -2.0), linewidth=0.5, color="black", linestyle="--")
    pyplot.plot([], linewidth=0.5, color="black", label="Slope $-1$")
    pyplot.plot([], linewidth=0.5, color="black", linestyle="--", label="Slope $-2$")
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlim(3.0, 768.0)
    pyplot.xlabel("$N$")
    pyplot.ylabel("Error")
pyplot.tight_layout()
pyplot.legend(loc="center right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure12.pgf")
pyplot.show()


# In[33]:


pyplot.figure(figsize=(8.0, 8.0))
for i, d in enumerate(d_list):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ L = {} $".format(d_name[i]))
    for j, theta in enumerate(theta_name):
        for l, mu in enumerate(mu_name):
            pyplot.plot(n_list, filter_array(norm2_list[str((d, l, j))], lambda x: x < 1.0e-1), label="$ \\theta = {}, {} $".format(theta, mu))
            pyplot.scatter(n_list, filter_array(norm2_list[str((d, l, j))], lambda x: x < 1.0e-1), s=2.0)
    if i == 0:
        pyplot.plot(*calc_log_line(4.0, 512.0, 5.0e-3, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(4.0, 512.0, 3.0e-4, -2.0), linewidth=0.5, color="black", linestyle="--")
    elif i == 1:
        pyplot.plot(*calc_log_line(16.0, 512.0, 0.15e-5, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(4.0, 512.0, 0.3e-7, -2.0), linewidth=0.5, color="black", linestyle="--")
    elif i == 2:
        pyplot.plot(*calc_log_line(64.0, 512.0, 1.0e-44, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(4.0, 512.0, 3.0e-45, -2.0), linewidth=0.5, color="black", linestyle="--")
    pyplot.plot([], linewidth=0.5, color="black", label="Slope $-1$")
    pyplot.plot([], linewidth=0.5, color="black", linestyle="--", label="Slope $-2$")
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlim(3.0, 768.0)
    pyplot.xlabel("$N$")
    pyplot.ylabel("Error")
pyplot.tight_layout()
pyplot.legend(loc="center right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure13.pgf")
pyplot.show()


# In[34]:


pyplot.figure(figsize=(8.0, 8.0))
for i, d in enumerate(d_list):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ L = {} $".format(d_name[i]))
    for j, theta in enumerate(theta_name):
        for l, mu in enumerate(mu_name):
            pyplot.plot(comp_list[str((d, l))], filter_array(normi_list[str((d, l, j))], lambda x: x < 1.0e-1), label="$ \\theta = {}, {} $".format(theta, mu))
            pyplot.scatter(comp_list[str((d, l))], filter_array(normi_list[str((d, l, j))], lambda x: x < 1.0e-1), s=2.0)
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("Estimated complexity")
    pyplot.ylabel("Error")
pyplot.tight_layout()
pyplot.legend(loc="center right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure14.pgf")
pyplot.show()


# In[35]:


pyplot.figure(figsize=(8.0, 8.0))
for i, d in enumerate(d_list):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ L = {} $".format(d_name[i]))
    for j, theta in enumerate(theta_name):
        for l, mu in enumerate(mu_name):
            pyplot.plot(comp_list[str((d, l))], filter_array(norm2_list[str((d, l, j))], lambda x: x < 1.0e-1), label="$ \\theta = {}, {} $".format(theta, mu))
            pyplot.scatter(comp_list[str((d, l))], filter_array(norm2_list[str((d, l, j))], lambda x: x < 1.0e-1), s=2.0)
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("Estimated complexity")
    pyplot.ylabel("Error")
pyplot.tight_layout()
pyplot.legend(loc="center right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure15.pgf")
pyplot.show()


# In[36]:


with shelve.open("Result") as db:
    d_list = db[str((2, 5, "d"))]
    d_name = db[str((2, 5, "d", "name"))]
    m_list = db[str((2, 5, "m"))]
    theta_name = db[str((2, 5, "theta", "name"))]
    normi_list = db[str((2, 5, "normi"))]
    norm2_list = db[str((2, 5, "norm2"))]


# In[37]:


pyplot.figure(figsize=(8.0, 8.0))
for i, d in enumerate(d_list):
    if i == 0:
        continue
    pyplot.subplot(2, 2, i)
    pyplot.title("$ L = {} $".format(d_name[i]))
    for j, theta in enumerate(theta_name):
        pyplot.plot(m_list, filter_array(normi_list[str((d, j))], lambda x: x < 1.0e-1), label="$ \\theta = {} $".format(theta))
        pyplot.scatter(m_list, filter_array(normi_list[str((d, j))], lambda x: x < 1.0e-1), s=2.0)
    if i == 1:
        pyplot.plot(*calc_log_line(10.0, 1600.0, 1.2e-2, -1.0), linewidth=0.5, color="black")
    elif i == 2:
        pyplot.plot(*calc_log_line(75.0, 12800.0, 1.8e-6, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(2.0, 75.0, 1.0e-3, -2.0), linewidth=0.5, color="black", linestyle="--")
    pyplot.plot([], linewidth=0.5, color="black", label="Slope $-1$")
    pyplot.plot([], linewidth=0.5, color="black", linestyle="--", label="Slope $-2$")
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("$ M / L $")
    pyplot.ylabel("Error")
pyplot.tight_layout()
pyplot.legend(loc="center right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure16.pgf")
pyplot.show()


# In[38]:


pyplot.figure(figsize=(8.0, 8.0))
for i, d in enumerate(d_list):
    if i == 0:
        continue
    pyplot.subplot(2, 2, i)
    pyplot.title("$ L = {} $".format(d_name[i]))
    for j, theta in enumerate(theta_name):
        pyplot.plot(m_list, filter_array(norm2_list[str((d, j))], lambda x: x < 1.0e-1), label="$ \\theta = {} $".format(theta))
        pyplot.scatter(m_list, filter_array(norm2_list[str((d, j))], lambda x: x < 1.0e-1), s=2.0)
    if i == 1:
        pyplot.plot(*calc_log_line(10.0, 1600.0, 0.8e-2, -1.0), linewidth=0.5, color="black")
    elif i == 2:
        pyplot.plot(*calc_log_line(75.0, 12800.0, 1.2e-6, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(2.0, 75.0, 0.7e-3, -2.0), linewidth=0.5, color="black", linestyle="--")
    pyplot.plot([], linewidth=0.5, color="black", label="Slope $-1$")
    pyplot.plot([], linewidth=0.5, color="black", linestyle="--", label="Slope $-2$")
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("$ M / L $")
    pyplot.ylabel("Error")
pyplot.tight_layout()
pyplot.legend(loc="center right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure17.pgf")
pyplot.show()


# In[39]:


with shelve.open("Result") as db:
    d_list = db[str((2, 6, "d"))]
    d_name = db[str((2, 6, "d", "name"))]
    mu_name = db[str((2, 6, "mu", "name"))]
    n_list = db[str((2, 6, "n"))]
    theta_name = db[str((2, 6, "theta", "name"))]
    normi_list = db[str((2, 6, "normi"))]
    norm2_list = db[str((2, 6, "norm2"))]
    comp_list = db[str((2, 6, "comp"))]


# In[40]:


pyplot.figure(figsize=(8.0, 8.0))
for i, d in enumerate(d_list):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ L = {} $".format(d_name[i]))
    for j, theta in enumerate(theta_name):
        for l, mu in enumerate(mu_name):
            pyplot.plot(n_list, filter_array(normi_list[str((d, l, j))], lambda x: x < 1.0e-1), label="$ \\theta = {}, {} $".format(theta, mu))
            pyplot.scatter(n_list, filter_array(normi_list[str((d, l, j))], lambda x: x < 1.0e-1), s=2.0)
    if i == 0:
        pyplot.plot(*calc_log_line(4.0, 512.0, 8.0e-4, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(4.0, 512.0, 1.0e-2, -0.0), linewidth=0.5, color="black", linestyle="--")
    elif i == 1:
        pyplot.plot(*calc_log_line(4.0, 512.0, 0.6e-5, -1.0), linewidth=0.5, color="black")
    elif i == 2:
        pyplot.plot(*calc_log_line(64.0, 512.0, 0.1e-43, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(4.0, 512.0, 1.0e-45, -2.0), linewidth=0.5, color="black", linestyle=":")
    pyplot.plot([], linewidth=0.5, color="black", label="Slope $-1$")
    pyplot.plot([], linewidth=0.5, color="black", linestyle="--", label="Slope $0$")
    pyplot.plot([], linewidth=0.5, color="black", linestyle=":", label="Slope $-2$")
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlim(3.0, 768.0)
    pyplot.xlabel("$N$")
    pyplot.ylabel("Error")
pyplot.tight_layout()
pyplot.legend(loc="center right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure18.pgf")
pyplot.show()


# In[41]:


pyplot.figure(figsize=(8.0, 8.0))
for i, d in enumerate(d_list):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ L = {} $".format(d_name[i]))
    for j, theta in enumerate(theta_name):
        for l, mu in enumerate(mu_name):
            pyplot.plot(n_list, filter_array(norm2_list[str((d, l, j))], lambda x: x < 1.0e-1), label="$ \\theta = {}, {} $".format(theta, mu))
            pyplot.scatter(n_list, filter_array(norm2_list[str((d, l, j))], lambda x: x < 1.0e-1), s=2.0)
    if i == 0:
        pyplot.plot(*calc_log_line(4.0, 512.0, 8.0e-4, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(4.0, 512.0, 0.7e-2, -0.5), linewidth=0.5, color="black", linestyle="--")
    elif i == 1:
        pyplot.plot(*calc_log_line(4.0, 512.0, 0.6e-5, -1.0), linewidth=0.5, color="black")
    elif i == 2:
        pyplot.plot(*calc_log_line(64.0, 512.0, 0.1e-43, -1.0), linewidth=0.5, color="black")
        pyplot.plot(*calc_log_line(4.0, 512.0, 1.0e-45, -2.0), linewidth=0.5, color="black", linestyle=":")
    pyplot.plot([], linewidth=0.5, color="black", label="Slope $-1$")
    pyplot.plot([], linewidth=0.5, color="black", linestyle="--", label="Slope $ -1 / 2 $")
    pyplot.plot([], linewidth=0.5, color="black", linestyle=":", label="Slope $-2$")
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlim(3.0, 768.0)
    pyplot.xlabel("$N$")
    pyplot.ylabel("Error")
pyplot.tight_layout()
pyplot.legend(loc="center right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure19.pgf")
pyplot.show()


# In[42]:


pyplot.figure(figsize=(8.0, 8.0))
for i, d in enumerate(d_list):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ L = {} $".format(d_name[i]))
    for j, theta in enumerate(theta_name):
        for l, mu in enumerate(mu_name):
            pyplot.plot(comp_list[str((d, l))], filter_array(normi_list[str((d, l, j))], lambda x: x < 1.0e-1), label="$ \\theta = {}, {} $".format(theta, mu))
            pyplot.scatter(comp_list[str((d, l))], filter_array(normi_list[str((d, l, j))], lambda x: x < 1.0e-1), s=2.0)
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("Estimated complexity")
    pyplot.ylabel("Error")
pyplot.tight_layout()
pyplot.legend(loc="center right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure20.pgf")
pyplot.show()


# In[43]:


pyplot.figure(figsize=(8.0, 8.0))
for i, d in enumerate(d_list):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ L = {} $".format(d_name[i]))
    for j, theta in enumerate(theta_name):
        for l, mu in enumerate(mu_name):
            pyplot.plot(comp_list[str((d, l))], filter_array(norm2_list[str((d, l, j))], lambda x: x < 1.0e-1), label="$ \\theta = {}, {} $".format(theta, mu))
            pyplot.scatter(comp_list[str((d, l))], filter_array(norm2_list[str((d, l, j))], lambda x: x < 1.0e-1), s=2.0)
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("Estimated complexity")
    pyplot.ylabel("Error")
pyplot.tight_layout()
pyplot.legend(loc="center right", bbox_to_anchor=(2.0, 0.5))
pyplot.savefig("Figure21.pgf")
pyplot.show()

