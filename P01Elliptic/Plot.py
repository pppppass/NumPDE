
# coding: utf-8

# In[1]:


import shelve
import numpy
import scipy.stats
import matplotlib
matplotlib.use("pgf")
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


# In[48]:


ana = numpy.load("Result1.npy")
sol = numpy.load("Result2.npy")


# In[49]:


n = 512
x, y = numpy.linspace(0.0, 1.0, n+1), numpy.linspace(0.0, 1.0, n+1)
y, x = numpy.meshgrid(x, y)


# In[50]:


pyplot.figure(figsize=(4.0, 4.0))
pyplot.subplot(projection="3d", aspect="equal")
pyplot.gca().view_init(25.0, -210.0)
pyplot.gca().plot_surface(x[::8, ::8], y[::8, ::8], ana[::8, ::8], cmap=matplotlib.cm.viridis).set_rasterized(True)
pyplot.xlabel("$x$")
pyplot.ylabel("$y$")
pyplot.gca().set_zlabel("$z$")
pyplot.savefig("Figure01.pgf", dpi=200.0)
pyplot.show()


# In[5]:


pyplot.figure(figsize=(8.0, 3.0))
pyplot.subplot(1, 2, 1)
pyplot.imshow(sol.transpose(), origin="lower")
pyplot.xlabel("$x$")
pyplot.ylabel("$y$")
pyplot.colorbar()
pyplot.subplot(1, 2, 2)
pyplot.imshow((sol - ana).transpose(), origin="lower")
pyplot.xlabel("$x$")
pyplot.ylabel("$y$")
pyplot.colorbar()
pyplot.tight_layout()
pyplot.savefig("Figure02.pgf")
pyplot.show()


# In[6]:


n_list = sorted(
      [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    + [3, 6, 11, 23, 45, 91, 181, 362, 724, 1448, 2896]
)
with shelve.open("Result") as db:
    data = {n: db[str((1, "error", n))] for n in n_list}


# In[7]:


err_list = [data[n][0] for n in n_list]
n_unif = numpy.logspace(numpy.log10(2.0), numpy.log10(4096.0), 50)
b, a, _, _, _ = scipy.stats.linregress(numpy.log(n_list), numpy.log(err_list))
pyplot.figure(figsize=(6.0, 4.0))
pyplot.plot(n_list, err_list, label="<LABEL1>")
pyplot.scatter(n_list, err_list, s=5.0)
pyplot.plot([2.0, 4096.0], [1.0e-5, 1.0e-5], color="black", linewidth=0.5, linestyle=":", label="Required")
pyplot.plot(n_unif, 3.0 / 2.0 * (3.0e-13 * n_unif**2 + 2.0 / 3.0 * numpy.pi**3 / n_unif**2), color="black", linewidth=0.5, label="Estimated")
pyplot.plot(n_list, numpy.exp(a + b * numpy.log(n_list)), color="black", linewidth=0.5, linestyle="--", label="Fit")
pyplot.legend()
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("$N$")
pyplot.ylabel("Error")
pyplot.savefig("Figure03.pgf")
pyplot.show()
with open("Text01.txt", "w") as f:
    f.write("\\log \\normi{{e_i}} = \\text{{{:.5f}}} + \\text{{{:.5f}}} \\log h^{{-1}}".format(a, b))


# In[8]:


n_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
with open("Table1.tbl", "w") as f:
    for n in n_list:
        f.write("{} & {:.5e} & {:.5f} & {} \\\\\n".format(n, *data[n]))
        f.write("\\hline\n")


# In[9]:


n_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
with shelve.open("Result") as db:
    data = {n: db[str((1, "order", n))] for n in n_list}


# In[11]:


err_list = [3.0 / 4.0 * data[n][0] for n in n_list]
n_unif = numpy.logspace(numpy.log10(2.0), numpy.log10(2048.0), 50)
b, a, _, _, _ = scipy.stats.linregress(numpy.log(n_list), numpy.log(err_list))
pyplot.figure(figsize=(6.0, 4.0))
pyplot.plot(n_list, err_list, label="<LABEL2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>")
pyplot.scatter(n_list, err_list, s=5.0)
pyplot.plot([2.0, 2048.0], [1.0e-5, 1.0e-5], color="black", linewidth=0.5, linestyle=":", label="Required")
pyplot.plot(n_list, numpy.exp(a + b * numpy.log(n_list)), color="black", linewidth=0.5, linestyle="--", label="Fit")
pyplot.legend()
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("$N$")
pyplot.ylabel("Error")
pyplot.savefig("Figure04.pgf")
pyplot.show()
with open("Text02.txt", "w") as f:
    f.write("\\log \\frac{{4}}{{3}} \\normi{{ U_h - U_{{ h / 2 }} }} = \\text{{{:.5f}}} + \\text{{{:.5f}}} \\log h^{{-1}}".format(a, b))
with open("Text03.txt", "w") as f:
    f.write("{:.5e}".format(numpy.exp(a)))


# In[13]:


n_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
with shelve.open("Result") as db:
    data = {n: db[str((1, "extrapolate", n))] for n in n_list}


# In[25]:


err_list = [data[n][0] for n in n_list]
n_unif = numpy.logspace(numpy.log10(2.0), numpy.log10(2048.0), 50)
b, a, _, _, _ = scipy.stats.linregress(numpy.log(n_list[1:-1]), numpy.log(err_list[1:-1]))
pyplot.figure(figsize=(6.0, 4.0))
pyplot.plot(n_list, err_list, label="<LABEL3~~~~~~~~>")
pyplot.scatter(n_list, err_list, s=5.0)
pyplot.plot(n_list, numpy.exp(a + b * numpy.log(n_list)), color="black", linewidth=0.5, linestyle="--", label="Fit")
pyplot.legend()
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("$N$")
pyplot.ylabel("Error")
pyplot.savefig("Figure05.pgf")
pyplot.show()
with open("Text04.txt", "w") as f:
    f.write("\\log \\normi{{U_h^1}} = \\text{{{:.5f}}} + \\text{{{:.5f}}} \\log h^{{-1}}".format(a, b))


# In[35]:


n_list = [2, 4, 8, 16, 32, 64, 128]#, 256, 512]
tol_list = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10]
with shelve.open("Result") as db:
    data = {n: [db[str((1, "tolerance", n, tol))][0] for tol in tol_list] for n in n_list}


# In[39]:


pyplot.figure(figsize=(6.0, 4.0))
for n in n_list:
    pyplot.plot(tol_list, data[n], label="$ n = {} $".format(n))
pyplot.legend(loc="upper left")
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("Tolerance")
pyplot.ylabel("Error")
pyplot.savefig("Figure06.pgf")
pyplot.show()


# In[52]:


ana = numpy.load("Result3.npy")
sol = numpy.load("Result4.npy")


# In[53]:


n = 512
x, y = numpy.linspace(0.0, 1.0, n+1), numpy.linspace(0.0, 1.0, n+1)
y, x = numpy.meshgrid(x, y)


# In[54]:


pyplot.figure(figsize=(4.0, 4.0))
pyplot.subplot(projection="3d", aspect="equal")
pyplot.gca().view_init(25.0, -65.0)
pyplot.gca().plot_surface(x[::8, ::8], y[::8, ::8], ana[::8, ::8], cmap=matplotlib.cm.viridis).set_rasterized(True)
pyplot.xlabel("$x$")
pyplot.ylabel("$y$")
pyplot.gca().set_zlabel("$z$")
pyplot.savefig("Figure07.pgf", dpi=200.0)
pyplot.show()


# In[55]:


pyplot.figure(figsize=(8.0, 3.0))
pyplot.subplot(1, 2, 1)
pyplot.imshow(sol.transpose(), origin="lower")
pyplot.xlabel("$x$")
pyplot.ylabel("$y$")
pyplot.colorbar()
pyplot.subplot(1, 2, 2)
pyplot.imshow((sol - ana).transpose(), origin="lower")
pyplot.xlabel("$x$")
pyplot.ylabel("$y$")
pyplot.colorbar()
pyplot.tight_layout()
pyplot.savefig("Figure08.pgf")
pyplot.show()


# In[58]:


n_list = sorted(
      [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    + [3, 6, 11, 23, 45, 91, 181, 362, 724, 1448, 2896]
)
with shelve.open("Result") as db:
    data = {n: db[str((2, "error", n))] for n in n_list}


# In[65]:


err_list = [data[n][0] for n in n_list]
n_unif = numpy.logspace(numpy.log10(3.0), numpy.log10(4096.0), 50)
b, a, _, _, _ = scipy.stats.linregress(numpy.log(n_list[3:-3]), numpy.log(err_list[3:-3]))
pyplot.figure(figsize=(6.0, 4.0))
pyplot.plot(n_list, err_list, label="<LABEL4>")
pyplot.scatter(n_list, err_list, s=5.0)
pyplot.plot([3.0, 4096.0], [1.0e-5, 1.0e-5], color="black", linewidth=0.5, linestyle=":", label="Required")
pyplot.plot(n_unif, 3.0 / 8.0 * (1.0e-11 * n_unif**2 + numpy.sqrt(2) / n_unif**2), color="black", linewidth=0.5, label="Estimated")
pyplot.plot(n_list, numpy.exp(a + b * numpy.log(n_list)), color="black", linewidth=0.5, linestyle="--", label="Fit")
pyplot.legend()
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("$N$")
pyplot.ylabel("Error")
pyplot.savefig("Figure09.pgf")
pyplot.show()
with open("Text05.txt", "w") as f:
    f.write("\\log \\normi{{e_i}} = \\text{{{:.5f}}} + \\text{{{:.5f}}} \\log h^{{-1}}".format(a, b))


# In[66]:


n_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
with open("Table2.tbl", "w") as f:
    for n in n_list:
        f.write("{} & {:.5e} & {:.5f} & {} \\\\\n".format(n, *data[n]))
        f.write("\\hline\n")


# In[69]:


n_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
with shelve.open("Result") as db:
    data = {n: db[str((2, "order", n))] for n in n_list}


# In[72]:


err_list = [3.0 / 4.0 * data[n][0] for n in n_list]
n_unif = numpy.logspace(numpy.log10(4.0), numpy.log10(2048.0), 50)
b, a, _, _, _ = scipy.stats.linregress(numpy.log(n_list[1:-1]), numpy.log(err_list[1:-1]))
pyplot.figure(figsize=(6.0, 4.0))
pyplot.plot(n_list, err_list, label="<LABEL5~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>")
pyplot.scatter(n_list, err_list, s=5.0)
pyplot.plot([4.0, 2048.0], [1.0e-5, 1.0e-5], color="black", linewidth=0.5, linestyle=":", label="Required")
pyplot.plot(n_list, numpy.exp(a + b * numpy.log(n_list)), color="black", linewidth=0.5, linestyle="--", label="Fit")
pyplot.legend()
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("$N$")
pyplot.ylabel("Error")
pyplot.savefig("Figure10.pgf")
pyplot.show()
with open("Text06.txt", "w") as f:
    f.write("\\log \\frac{{4}}{{3}} \\normi{{ U_h - U_{{ h / 2 }} }} = \\text{{{:.5f}}} + \\text{{{:.5f}}} \\log h^{{-1}}".format(a, b))
with open("Text07.txt", "w") as f:
    f.write("{:.5e}".format(numpy.exp(a)))


# In[73]:


n_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
with shelve.open("Result") as db:
    data = {n: db[str((2, "extrapolate", n))] for n in n_list}


# In[75]:


err_list = [data[n][0] for n in n_list]
n_unif = numpy.logspace(numpy.log10(4.0), numpy.log10(2048.0), 50)
b, a, _, _, _ = scipy.stats.linregress(numpy.log(n_list[1:-3]), numpy.log(err_list[1:-3]))
pyplot.figure(figsize=(6.0, 4.0))
pyplot.plot(n_list, err_list, label="<LABEL6~~~~~~~~>")
pyplot.scatter(n_list, err_list, s=5.0)
pyplot.plot(n_list, numpy.exp(a + b * numpy.log(n_list)), color="black", linewidth=0.5, linestyle="--", label="Fit")
pyplot.legend()
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("$N$")
pyplot.ylabel("Error")
pyplot.savefig("Figure11.pgf")
pyplot.show()
with open("Text08.txt", "w") as f:
    f.write("\\log \\normi{{U_h^1}} = {:.5f} + {:.5f} \\log h^{{-1}}".format(a, b))


# In[78]:


n_list = [4, 8, 16, 32, 64, 128]#, 256, 512]
tol_list = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10]
with shelve.open("Result") as db:
    data = {n: [db[str((2, "tolerance", n, tol))][0] for tol in tol_list] for n in n_list}


# In[79]:


pyplot.figure(figsize=(6.0, 4.0))
for n in n_list:
    pyplot.plot(tol_list, data[n], label="$ n = {} $".format(n))
pyplot.legend(loc="upper left")
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("Tolerance")
pyplot.ylabel("Error")
pyplot.savefig("Figure12.pgf")
pyplot.show()


# In[80]:


ana = numpy.load("Result5.npy")
sol = numpy.load("Result6.npy")


# In[81]:


n = 512
x, y = numpy.linspace(0.0, 1.0, n+1), numpy.linspace(0.0, 1.0, n+1)
y, x = numpy.meshgrid(x, y)


# In[86]:


pyplot.figure(figsize=(4.0, 4.0))
pyplot.subplot(projection="3d", aspect="equal")
pyplot.gca().view_init(25.0, 20.0)
pyplot.gca().plot_surface(x[::8, ::8], y[::8, ::8], ana[::8, ::8], cmap=matplotlib.cm.viridis).set_rasterized(True)
pyplot.xlabel("$x$")
pyplot.ylabel("$y$")
pyplot.gca().set_zlabel("$z$")
pyplot.savefig("Figure13.pgf", dpi=200.0)
pyplot.show()


# In[87]:


pyplot.figure(figsize=(8.0, 3.0))
pyplot.subplot(1, 2, 1)
pyplot.imshow(sol.transpose(), origin="lower")
pyplot.xlabel("$x$")
pyplot.ylabel("$y$")
pyplot.colorbar()
pyplot.subplot(1, 2, 2)
pyplot.imshow((sol - ana).transpose(), origin="lower")
pyplot.xlabel("$x$")
pyplot.ylabel("$y$")
pyplot.colorbar()
pyplot.tight_layout()
pyplot.savefig("Figure14.pgf")
pyplot.show()


# In[114]:


n_list = sorted(
      [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    + [3, 6, 11, 23, 45, 91, 181, 362, 724, 1448, 2896]
)
with shelve.open("Result") as db:
    data = {n: db[str((3, "error", n))] for n in n_list}


# In[115]:


err_list = [data[n][0] for n in n_list]
n_unif = numpy.logspace(numpy.log10(2.0), numpy.log10(4096.0), 50)
b, a, _, _, _ = scipy.stats.linregress(numpy.log(n_list[4:-3]), numpy.log(err_list[4:-3]))
pyplot.figure(figsize=(6.0, 4.0))
pyplot.plot(n_list, err_list, label="<LABEL7>")
pyplot.scatter(n_list, err_list, s=5.0)
pyplot.plot([2.0, 4096.0], [1.0e-5, 1.0e-5], color="black", linewidth=0.5, linestyle=":", label="Required")
pyplot.plot(n_unif, 15.0 / 8.0 * (1.0e-11 * n_unif**2 + 2.0 / 3.0 / numpy.pi / n_unif**2), color="black", linewidth=0.5, label="Estimated")
pyplot.plot(n_list, numpy.exp(a + b * numpy.log(n_list)), color="black", linewidth=0.5, linestyle="--", label="Fit")
pyplot.legend()
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("$N$")
pyplot.ylabel("Error")
pyplot.savefig("Figure15.pgf")
pyplot.show()
with open("Text09.txt", "w") as f:
    f.write("\\log \\normi{{e_i}} = \\text{{{:.5f}}} + \\text{{{:.5f}}} \\log h^{{-1}}".format(a, b))


# In[99]:


n_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
with open("Table3.tbl", "w") as f:
    for n in n_list:
        f.write("{} & {:.5e} & {:.5f} & {} \\\\\n".format(n, *data[n]))
        f.write("\\hline\n")


# In[102]:


n_list = [3, 6, 12, 24, 48, 96, 192, 384, 768, 1536]#[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
with shelve.open("Result") as db:
    data = {n: db[str((3, "order", n))] for n in n_list}


# In[104]:


err_list = [3.0 / 4.0 * data[n][0] for n in n_list]
n_unif = numpy.logspace(numpy.log10(2.0), numpy.log10(2048.0), 50)
b, a, _, _, _ = scipy.stats.linregress(numpy.log(n_list[2:-1]), numpy.log(err_list[2:-1]))
pyplot.figure(figsize=(6.0, 4.0))
pyplot.plot(n_list, err_list, label="<LABEL8~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>")
pyplot.scatter(n_list, err_list, s=5.0)
pyplot.plot([2.0, 2048.0], [1.0e-5, 1.0e-5], color="black", linewidth=0.5, linestyle=":", label="Required")
pyplot.plot(n_list, numpy.exp(a + b * numpy.log(n_list)), color="black", linewidth=0.5, linestyle="--", label="Fit")
pyplot.legend()
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("$N$")
pyplot.ylabel("Error")
pyplot.savefig("Figure16.pgf")
pyplot.show()
with open("Text10.txt", "w") as f:
    f.write("\\log \\frac{{4}}{{3}} \\normi{{ U_h - U_{{ h / 2 }} }} = \\text{{{:.5f}}} + \\text{{{:.5f}}} \\log h^{{-1}}".format(a, b))
with open("Text11.txt", "w") as f:
    f.write("{:.5e}".format(numpy.exp(a)))


# In[106]:


n_list = [3, 6, 12, 24, 48, 96, 192, 384, 768, 1536]#[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
with shelve.open("Result") as db:
    data = {n: db[str((3, "extrapolate", n))] for n in n_list}


# In[108]:


err_list = [data[n][0] for n in n_list]
n_unif = numpy.logspace(numpy.log10(2.0), numpy.log10(2048.0), 50)
b, a, _, _, _ = scipy.stats.linregress(numpy.log(n_list), numpy.log(err_list))
pyplot.figure(figsize=(6.0, 4.0))
pyplot.plot(n_list, err_list, label="<LABEL9~~~~~~~~>")
pyplot.scatter(n_list, err_list, s=5.0)
pyplot.plot(n_list, numpy.exp(a + b * numpy.log(n_list)), color="black", linewidth=0.5, linestyle="--", label="Fit")
pyplot.legend()
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("$N$")
pyplot.ylabel("Error")
pyplot.savefig("Figure17.pgf")
pyplot.show()
with open("Text12.txt", "w") as f:
    f.write("\\log \\normi{{U_h^1}} = {:.5f} + {:.5f} \\log h^{{-1}}".format(a, b))


# In[111]:


n_list = [2, 4, 8, 16, 32, 64, 128]#, 256, 512]
tol_list = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10]
with shelve.open("Result") as db:
    data = {n: [db[str((3, "tolerance", n, tol))][0] for tol in tol_list] for n in n_list}


# In[112]:


pyplot.figure(figsize=(6.0, 4.0))
for n in n_list:
    pyplot.plot(tol_list, data[n], label="$ n = {} $".format(n))
pyplot.legend(loc="upper left")
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("Tolerance")
pyplot.ylabel("Error")
pyplot.savefig("Figure18.pgf")
pyplot.show()

