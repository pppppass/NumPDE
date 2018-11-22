
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
import models
import exts


# In[2]:


w, d = 1.0, 1000.0
n = 100
h = w / n


# In[3]:


theta_list = [0.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0, 3.0 / 4.0, 1.0]
theta_name = ["0", " 1 / 4 ", " 1 / 3 ", " 1 / 2 ", " 2 / 3 ", " 3 / 4 ", "1"]
m_list = numpy.logspace(0.0, 5.0, 41) * d


# In[4]:


t_chk = [1.0, 10.0, 100.0, 1000.0]
t_name = [1, 10, 100, 1000]


# In[5]:


norm_2_mask = numpy.ones(n+1)
norm_2_mask[[0, -1]] /= numpy.sqrt(2)


# In[6]:


rt = [{}, {}]


# In[7]:


for theta in theta_list:
    
    for t in t_chk:
        rt[0][(t, theta)] = []
        rt[1][(t, theta)] = []
    
    for m in m_list:
        
        m = int(m + 0.5)
        p =  (n * m - 1) // 150000000 + 1
        tau = d / m
        
        x = numpy.linspace(0.0, w, n+1)[:, None]
        ldu = m // p + 2
        u = numpy.zeros((n+1, ldu))
        u[:, 0] = models.calc_ana_sol_3(x, 0.0)[:, 0]
        
        t_ctr = 0
        
        print("theta = {}, m = {} started, {} chunks".format(theta, m, p))
        
        for k in range(p):
            
            m_l, m_u = (k * m) // p, ((k+1) * m) // p
            t = numpy.linspace(tau * m_l, tau * m_u, m_u - m_l + 1)[None, :]
            
            a = models.calc_coef_3(x, t)
            f = models.calc_sour_3(x, t)
            alpha = models.calc_alpha_3(t)
            g = models.calc_grad_3(t)
            
            exts.para_theta_ghost_full_wrapper(n, m_u - m_l, w, tau * (m_u - m_l), a, f, alpha, g, theta, u, ldu)
            
            while t_ctr < len(t_chk):
                t_ = t_chk[t_ctr]
                if t_ > tau * m_u + 1.0e-10:
                    break
                m_ = int(t_ / tau)
                u_m_sol = u[:, m_ - m_l]
                u_m_ana = models.calc_ana_sol_3(x, t[0, m_ - m_l])[:, 0]
                rt[0][(t_, theta)].append(numpy.linalg.norm(u_m_sol - u_m_ana, numpy.infty))
                rt[1][(t_, theta)].append(numpy.linalg.norm((u_m_sol - u_m_ana) * norm_2_mask, 2.0) * numpy.sqrt(h))
                t_ctr += 1
            
            u[:, 0] = u[:, m_u - m_l]
        
        del(u_m_sol)
        del(u_m_ana)
        del(u)
        del(f)
        del(a)


# In[8]:


with shelve.open("Result") as db:
    db[str((2, 1, "t"))] = t_chk
    db[str((2, 1, "t", "name"))] = t_name
    db[str((2, 1, "theta"))] = theta_list
    db[str((2, 1, "theta", "name"))] = theta_name
    db[str((2, 1, "m"))] = m_list
    db[str((2, 1, "normi"))] = rt[0]
    db[str((2, 1, "norm2"))] = rt[1]

