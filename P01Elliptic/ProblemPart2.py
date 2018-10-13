
# coding: utf-8

# In[12]:


import time
import shelve
import numpy
import scipy.sparse
import exts


# In[2]:


def get_ana_sol(size):
    n = size
    h = 1.0 / n
    g = numpy.linspace(0.0, 1.0, n+1)
    x, y = g[:, None], g[None, :]
    ana = numpy.log((x + 1.0)**2 + y**2) / 2.0 + numpy.arctan(y / (x + 1.0))
    return ana


# In[3]:


def solve_sol(size, tol, max_=50000):
    
    n = size
    h = 1.0 / n
    
    data = numpy.zeros((5, n+1, n-1))
    data[0, :, :] = 4.0
    data[0, [0, -1], :] = 2.0
    data[1, :, :] = -1.0
    data[1, [0, -1], :] = -1.0 / 2.0
    data[1, :, 0] = 0.0
    data[2, :, :] = -1.0
    data[3, :, :] = -1.0
    data[3, [0, -1], :] = -1.0 / 2.0
    data[3, :, -1] = 0.0
    data[4, :, :] = -1.0
    
    mat = scipy.sparse.dia_matrix((data.reshape(5, -1), [0, 1, n-1, -1, -n+1]), ((n+1)*(n-1), (n+1)*(n-1))).tocsr()
    del(data)
    
    g = numpy.linspace(0.0, 1.0, n+1)
    x, y = g[:, None], g[None, :]
    bdry1 = (1.0 - g) / (1.0 + g**2)
    bdry2 = numpy.log(g + 1.0)
    bdry3 = (2.0 - g) / (4.0 + g**2)
    bdry4 = numpy.log((g + 1.0)**2 + 1) / 2.0 + numpy.pi / 2.0 - numpy.arctan(g + 1.0)
    
    sol = numpy.zeros((n+1, n+1))
    sol[:, 0] = bdry2
    sol[:, -1] = bdry4
    
    vec = numpy.zeros((n+1, n-1))
    vec[0, :] -= h * bdry1[1:-1]
    vec[-1, :] += h * bdry3[1:-1]
    vec[1:-1, 0] += bdry2[1:-1]
    vec[0::n, 0] += bdry2[0::n] / 2.0
    vec[1:-1, -1] += bdry4[1:-1]
    vec[0::n, -1] += bdry4[0::n] / 2.0
    
    start = time.time()
    ctr = exts.solve_cg_infty_wrapper((n-1)*(n+1), mat.data, mat.indices, mat.indptr, vec, sol[:, 1:-1], tol, max_)
    end = time.time()
    
    return sol, end - start, ctr


# In[4]:


res = [[], [], [], []]
n_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
tol = 1.0e-11


# In[5]:


sol_old = None
for n in n_list:
    
    ana = get_ana_sol(n)
    
    sol, elap, ctr = solve_sol(n, tol)
    print("n = {} solved, {:.5f} seconds and {} iterations".format(n, elap, ctr))
    err = numpy.linalg.norm((sol - ana).flatten(), numpy.infty)
    print("Error is {:.5e}".format(err))
    
    res[0].append((n, err, elap, ctr))
    
    if sol_old is not None:
        
        ana_old = get_ana_sol(n//2)
        sol_ext = (4.0 * sol[::2, ::2] - sol_old) / 3.0
        err_ext = numpy.linalg.norm((sol_ext - ana_old).flatten(), numpy.infty)
        print("Extrapolated error is {:.5e}".format(err_ext))
        res[1].append((n//2, err_ext))
        del(sol_ext)
        del(ana_old)
    
        inc = sol[::2, ::2] - sol_old
        inc_norm = numpy.linalg.norm(inc.flatten(), numpy.infty)
        res[2].append((n//2, inc_norm))
        del(inc)
    
    del(sol_old)
    sol_old = sol
    del(sol)
    del(ana)

del(sol_old)


# In[6]:


n_list = [3, 6, 11, 23, 45, 91, 181, 362, 724, 1448, 2896]


# In[7]:


for n in n_list:
    
    ana = get_ana_sol(n)
    
    sol, elap, ctr = solve_sol(n, tol)
    print("n = {} solved, {:.5f} seconds and {} iterations".format(n, elap, ctr))
    err = numpy.linalg.norm((sol - ana).flatten(), numpy.infty)
    
    res[0].append((n, err, elap, ctr))

    del(sol)
    del(ana)


# In[8]:


n_list = [4, 8, 16, 32, 64, 128, 256, 512]
tol_list = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10]


# In[9]:


for n in n_list:
    ana = get_ana_sol(n)
    for tol in tol_list:
        sol, elap, ctr = solve_sol(n, tol)
        print("tol = {:.1e} solved, {:.5f} seconds and {} iterations".format(tol, elap, ctr))
        err = numpy.linalg.norm((sol - ana).flatten(), numpy.infty)
        res[3].append((n, tol, err, elap, ctr))
        del(sol)
    print("n = {} finished".format(n))
    del(ana)


# In[13]:


with shelve.open("Result") as db:
    for e in res[0]:
        db[str((2, "error", e[0]))] = e[1:]
    for e in res[1]:
        db[str((2, "extrapolate", e[0]))] = e[1:]
    for e in res[2]:
        db[str((2, "order", e[0]))] = e[1:]
    for e in res[3]:
        db[str((2, "tolerance", e[0], e[1]))] = e[2:]


# In[14]:


n = 512
tol = 1.0e-11


# In[15]:


ana = get_ana_sol(n)
sol, _, _ = solve_sol(n, tol)


# In[16]:


numpy.save("Result3.npy", ana)
numpy.save("Result4.npy", sol)

