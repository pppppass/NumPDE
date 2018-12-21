import numpy
import scipy.sparse
from utils import exts


def get_mat_tri(size, beta):
    n = size
    h = 1.0 / n
    data = numpy.zeros((5, n, n+1))
    data[0, :, :] = 4.0
    data[0, :, [0, -1]] = 2.0 + 2.0/3.0 * h * beta[[0, 2], None]
    data[0, -1, :] = 2.0 + 2.0/3.0 * h * beta[1]
    data[0, -1, [0, -1]] = 1.0 + 1.0/3.0 * h * beta[[0, 2]] + 1.0/3.0 * h * beta[1]
    data[1, :, :] = -1.0
    data[1, -1, :] = -1.0/2.0 + 1.0/6.0 * h * beta[1]
    data[1, :, 0] = 0.0
    data[2, :, :] = -1.0
    data[2, :, [0, -1]] = -1.0/2.0 + 1.0/6.0 * h * beta[[0, 2], None]
    data[2, 0, :] = 0.0
    data[3, :, :] = -1.0
    data[3, -1, :] = -1.0/2.0 + 1.0/6.0 * h * beta[1]
    data[3, :, -1] = 0.0
    data[4, :, :] = -1.0
    data[4, :, [0, -1]] = -1.0/2.0 + 1.0/6.0 * h * beta[[0, 2], None]
    data[4, -1, :] = 0.0
    a = scipy.sparse.dia_matrix((data.reshape(5, -1), [0, 1, n+1, -1, -n-1]), (n*(n+1), n*(n+1))).tocsr()
    return a


def get_mat_rect(size, beta):
    n = size
    h = 1.0 / n
    data = numpy.zeros((9, n, n+1))
    data[0, :, :] = 8.0/3.0
    data[0, :, [0, -1]] = 4.0/3.0 + 2.0/3.0 * h * beta[[0, 2], None]
    data[0, -1, :] = 4.0/3.0 + 2.0/3.0 * h * beta[1]
    data[0, -1, [0, -1]] = 2.0/3.0 + 1.0/3.0 * h * beta[[0, 2]] + 1.0/3.0 * h * beta[1]
    data[1, :, :] = -1.0/3.0
    data[1, -1, :] = -1.0/6.0 + 1.0/6.0 * h * beta[1]
    data[1, :, 0] = 0.0
    data[2, :, :] = -1.0/3.0
    data[2, 0, :], data[2, :, -1] = 0.0, 0.0
    data[3, :, :] = -1.0/3.0
    data[3, :, [0, -1]] = -1.0/6.0 + 1.0/6.0 * h * beta[[0, 2], None]
    data[3, 0, :] = 0.0
    data[4, :, :] = -1.0/3.0
    data[4, 0, :], data[4, :, 0] = 0.0, 0.0
    data[5, :, :] = -1.0/3.0
    data[5, -1, :] = -1.0/6.0 + 1.0/6.0 * h * beta[1]
    data[5, :, -1] = 0.0
    data[6, :, :] = -1.0/3.0
    data[6, -1, :], data[6, :, 0] = 0.0, 0.0
    data[7, :, :] = -1.0/3.0
    data[7, :, [0, -1]] = -1.0/6.0 + 1.0/6.0 * h * beta[[0, 2], None]
    data[7, -1, :] = 0.0
    data[8, :, :] = -1.0/3.0
    data[8, -1, :], data[8, :, -1] = 0.0, 0.0
    a = scipy.sparse.dia_matrix((data.reshape(9, -1), [0, 1, n, n+1, n+2, -1, -n, -n-1, -n-2]), (n*(n+1), n*(n+1))).tocsr()
    return a


def get_rhs_tri(size, beta, cond, sol=None):
    
    n, g1, g2, g3, g4, f, u = size, *cond, sol
    h = 1.0 / n
    
    t = numpy.linspace(0.0, 1.0, n+1)
    x, y = t[:, None], t[None, :]
    t2 = numpy.linspace(0.0, 1.0, 2*n+1)
    x2, y2 = t2[:, None], t2[None, :]

    r = numpy.zeros((n, n+1))

    b1 = g1(y)
    r[0, 1:-1] += b1[0, 1:-1]
    r[0, [0, -1]] += b1[0, [0, -1]] * (1.0/2.0 - 1.0/6.0 * h * beta[[0, 2]])
    
    if u is not None:
        u[0, :] = b1[0, :]

    b2 = g2(x2)
    r[:, 0] += h * (1.0/3.0 * b2[1::2, 0] + 1.0/6.0 * b2[2::2, 0])
    r[:-1, 0] += h * (1.0/3.0 * b2[3::2, 0] + 1.0/6.0 * b2[2:-1:2, 0])

    b3 = g3(y2)
    r[-1, :-1] += h * (1.0/3.0 * b3[0, 1::2] + 1.0/6.0 * b3[0, :-1:2])
    r[-1, 1:] += h * (1.0/3.0 * b3[0, 1::2] + 1.0/6.0 * b3[0, 2::2])

    b4 = g4(x2)
    r[:, -1] += h * (1.0/3.0 * b4[1::2, 0] + 1.0/6.0 * b4[2::2, 0])
    r[:-1, -1] += h * (1.0/3.0 * b4[3::2, 0] + 1.0/6.0 * b4[2:-1:2, 0])

    i = f(x2, y2)
    r[:, 1:] += h**2 / 2.0 * (1.0/6.0 * i[1::2, 2::2] + 1.0/6.0 * i[1::2, 1::2])
    r[:, 1:] += h**2 / 2.0 * (1.0/6.0 * i[1::2, 1::2] + 1.0/6.0 * i[2::2, 1::2])
    r[:-1, 1:] += h**2 / 2.0 * (1.0/6.0 * i[2:-1:2, 1::2] + 1.0/6.0 * i[3::2, 2::2])
    r[:-1, :-1] += h**2 / 2.0 * (1.0/6.0 * i[3::2, :-1:2] + 1.0/6.0 * i[3::2, 1::2])
    r[:-1, :-1] += h**2 / 2.0 * (1.0/6.0 * i[3::2, 1::2] + 1.0/6.0 * i[2:-1:2, 1::2])
    r[:, :-1] += h**2 / 2.0 * (1.0/6.0 * i[2::2, 1::2] + 1.0/6.0 * i[1::2, :-1:2])
    
    return r


def get_rhs_rect(size, beta, cond, sol=None):
    
    n, g1, g2, g3, g4, f, u = size, *cond, sol
    h = 1.0 / n
    
    t = numpy.linspace(0.0, 1.0, n+1)
    x, y = t[:, None], t[None, :]
    t2 = numpy.linspace(0.0, 1.0, 2*n+1)
    x2, y2 = t2[:, None], t2[None, :]

    r = numpy.zeros((n, n+1))

    b1 = g1(y)
    r[0, 1:-1] += (b1[0, 1:-1] + b1[0, :-2] + b1[0, 2:]) / 3.0
    r[0, [0, -1]] += b1[0, [0, -1]] * (1.0/6.0 - 1.0/6.0 * h * beta[[0, 2]]) + b1[0, [1, -2]] / 3.0
    
    if u is not None:
        u[0, :] = b1[0, :]

    b2 = g2(x2)
    r[:, 0] += h * (1.0/3.0 * b2[1::2, 0] + 1.0/6.0 * b2[2::2, 0])
    r[:-1, 0] += h * (1.0/3.0 * b2[3::2, 0] + 1.0/6.0 * b2[2:-1:2, 0])

    b3 = g3(y2)
    r[-1, :-1] += h * (1.0/3.0 * b3[0, 1::2] + 1.0/6.0 * b3[0, :-1:2])
    r[-1, 1:] += h * (1.0/3.0 * b3[0, 1::2] + 1.0/6.0 * b3[0, 2::2])

    b4 = g4(x2)
    r[:, -1] += h * (1.0/3.0 * b4[1::2, 0] + 1.0/6.0 * b4[2::2, 0])
    r[:-1, -1] += h * (1.0/3.0 * b4[3::2, 0] + 1.0/6.0 * b4[2:-1:2, 0])

    i = f(x2, y2)
    r[:-1, :-1] += h**2 * (
          1.0/9.0 * i[3::2, 1::2]
        + 1.0/18.0 * i[2:-1:2, 1::2]
        + 1.0/36.0 * i[2:-1:2, :-1:2]
        + 1.0/18.0 * i[3::2, :-1:2]
    )
    r[:, :-1] += h**2 * (
          1.0/18.0 * i[2::2, 1::2]
        + 1.0/9.0 * i[1::2, 1::2]
        + 1.0/18.0 * i[1::2, :-1:2]
        + 1.0/36.0 * i[2::2, :-1:2]
    )
    r[:, 1:] += h**2 * (
          1.0/36.0 * i[2::2, 2::2]
        + 1.0/18.0 * i[1::2, 2::2]
        + 1.0/9.0 * i[1::2, 1::2]
        + 1.0/18.0 * i[2::2, 1::2]
    )
    r[:-1, 1:] += h**2 * (
          1.0/18.0 * i[3::2, 2::2]
        + 1.0/36.0 * i[2:-1:2, 2::2]
        + 1.0/18.0 * i[2:-1:2, 1::2]
        + 1.0/9.0 * i[3::2, 1::2]
    )
    
    return r


def driver_fem_tri(size, beta, cond, eps=1.0e-11, iters=10000):
    n = size
    a = get_mat_tri(n, beta)
    u = numpy.zeros((n+1, n+1))
    r = get_rhs_tri(n, beta, cond, u)
    ctr = exts.solve_cg_infty_wrapper(n*(n+1), a.data, a.indices, a.indptr, r, u[1:, :], eps, iters)
    return u, ctr


def driver_fem_rect(size, beta, cond, eps=1.0e-11, iters=10000):
    n = size
    a = get_mat_rect(n, beta)
    u = numpy.zeros((n+1, n+1))
    r = get_rhs_rect(n, beta, cond, u)
    ctr = exts.solve_cg_infty_wrapper(n*(n+1), a.data, a.indices, a.indptr, r, u[1:, :], eps, iters)
    return u, ctr


def calc_int_tri(sol):
    u_sol = sol
    n = u_sol.shape[0] - 1
    u_sol2 = numpy.zeros((2*n+1, 2*n+1))
    u_sol2[::2, ::2] = u_sol
    u_sol2[::2, 1::2] = (u_sol[:, :-1] + u_sol[:, 1:]) / 2.0
    u_sol2[1::2, ::2] = (u_sol[:-1, :] + u_sol[1:, :]) / 2.0
    u_sol2[1::2, 1::2] = (u_sol[:-1, :-1] + u_sol[1:, 1:]) / 2.0
    return u_sol2


def calc_int_rect(sol):
    u_sol = sol
    n = u_sol.shape[0] - 1
    u_sol2 = numpy.zeros((2*n+1, 2*n+1))
    u_sol2[::2, ::2] = u_sol
    u_sol2[::2, 1::2] = (u_sol[:, :-1] + u_sol[:, 1:]) / 2.0
    u_sol2[1::2, ::2] = (u_sol[:-1, :] + u_sol[1:, :]) / 2.0
    u_sol2[1::2, 1::2] = (u_sol[:-1, :-1] + u_sol[:-1, 1:] + u_sol[1:, :-1] + u_sol[1:, 1:]) / 4.0
    return u_sol2
    

def calc_err_tri(sol, ana2, ana_x2, ana_y2):
    
    u_sol, u_ana2, u_ana_x2, u_ana_y2 = sol, ana2, ana_x2, ana_y2
    n = u_sol.shape[0] - 1
    h = 1.0 / n
    
    u_sol2 = numpy.zeros((2*n+1, 2*n+1))
    u_sol2[::2, ::2] = u_sol
    u_sol2[::2, 1::2] = (u_sol[:, :-1] + u_sol[:, 1:]) / 2.0
    u_sol2[1::2, ::2] = (u_sol[:-1, :] + u_sol[1:, :]) / 2.0
    u_sol2[1::2, 1::2] = (u_sol[:-1, :-1] + u_sol[1:, 1:]) / 2.0

    e_linf = numpy.linalg.norm((u_sol2 - u_ana2).flat, numpy.infty)
    
    e_l2 = numpy.sqrt(1.0/2.0 * h**2 * 1.0/3.0 * (
          ((u_ana2[2::2, 1::2] - u_sol2[2::2, 1::2])**2).sum()
        + ((u_ana2[1::2, 1::2] - u_sol2[1::2, 1::2])**2).sum()
        + ((u_ana2[:-1:2, 1::2] - u_sol2[:-1:2, 1::2])**2).sum()
        + ((u_ana2[1::2, 2::2] - u_sol2[1::2, 2::2])**2).sum()
        + ((u_ana2[:-1:2, 1::2] - u_sol2[:-1:2, 1::2])**2).sum()
        + ((u_ana2[1::2, 1::2] - u_sol2[1::2, 1::2])**2).sum()
    ))
    
    u_sol_x = (u_sol[1:, :] - u_sol[:-1, :]) / h
    u_sol_y = (u_sol[:, 1:] - u_sol[:, :-1]) / h
    
    e_h1 = numpy.sqrt(1.0/2.0 * h**2 * 1.0/3.0 * (
          ((u_ana_x2[2::2, 1::2] - u_sol_x[:, :-1])**2).sum()
        + ((u_ana_x2[1::2, 1::2] - u_sol_x[:, :-1])**2).sum()
        + ((u_ana_x2[:-1:2, 1::2] - u_sol_x[:, :-1])**2).sum()
        + ((u_ana_x2[1::2, 2::2] - u_sol_x[:, 1:])**2).sum()
        + ((u_ana_x2[:-1:2, 1::2] - u_sol_x[:, 1:])**2).sum()
        + ((u_ana_x2[1::2, 1::2] - u_sol_x[:, 1:])**2).sum()
        + ((u_ana_y2[2::2, 1::2] - u_sol_y[1:, :])**2).sum()
        + ((u_ana_y2[1::2, 1::2] - u_sol_y[1:, :])**2).sum()
        + ((u_ana_y2[:-1:2, 1::2] - u_sol_y[1:, :])**2).sum()
        + ((u_ana_y2[1::2, 2::2] - u_sol_y[:-1, :])**2).sum()
        + ((u_ana_y2[:-1:2, 1::2] - u_sol_y[:-1, :])**2).sum()
        + ((u_ana_y2[1::2, 1::2] - u_sol_y[:-1, :])**2).sum()
    ))
    
    return e_linf, e_l2, e_h1


def calc_err_rect(sol, ana2, ana_x2, ana_y2):
    
    u_sol, u_ana2, u_ana_x2, u_ana_y2 = sol, ana2, ana_x2, ana_y2
    n = u_sol.shape[0] - 1
    h = 1.0 / n
    
    u_sol2 = numpy.zeros((2*n+1, 2*n+1))
    u_sol2[::2, ::2] = u_sol
    u_sol2[::2, 1::2] = (u_sol[:, :-1] + u_sol[:, 1:]) / 2.0
    u_sol2[1::2, ::2] = (u_sol[:-1, :] + u_sol[1:, :]) / 2.0
    u_sol2[1::2, 1::2] = (u_sol[:-1, :-1] + u_sol[:-1, 1:] + u_sol[1:, :-1] + u_sol[1:, 1:]) / 4.0

    e_linf = numpy.linalg.norm((u_sol2 - u_ana2).flat, numpy.infty)
    
    e_l2 = numpy.sqrt(h**2 * (
          1.0/36.0 * ((u_ana2[:-1:2, :-1:2] - u_sol2[:-1:2, :-1:2])**2).sum()
        + 1.0/9.0 * ((u_ana2[:-1:2, 1::2] - u_sol2[:-1:2, 1::2])**2).sum()
        + 1.0/36.0 * ((u_ana2[:-1:2, 2::2] - u_sol2[:-1:2, 2::2])**2).sum()
        + 1.0/9.0 * ((u_ana2[1::2, :-1:2] - u_sol2[1::2, :-1:2])**2).sum()
        + 4.0/9.0 * ((u_ana2[1::2, 1::2] - u_sol2[1::2, 1::2])**2).sum()
        + 1.0/9.0 * ((u_ana2[1::2, 2::2] - u_sol2[1::2, 2::2])**2).sum()
        + 1.0/36.0 * ((u_ana2[2::2, :-1:2] - u_sol2[2::2, :-1:2])**2).sum()
        + 1.0/9.0 * ((u_ana2[2::2, 1::2] - u_sol2[2::2, 1::2])**2).sum()
        + 1.0/36.0 * ((u_ana2[2::2, 2::2] - u_sol2[2::2, 2::2])**2).sum()
    ))
    
    u_sol_x = (u_sol[1:, :] - u_sol[:-1, :]) / h
    u_sol_y = (u_sol[:, 1:] - u_sol[:, :-1]) / h
    
    e_h1 = numpy.sqrt(1.0/2.0 * h**2 * 1.0/3.0 * (
          1.0/36.0 * ((u_ana_x2[:-1:2, :-1:2] - u_sol_x[:, :-1])**2).sum()
        + 1.0/9.0 * ((u_ana_x2[:-1:2, 1::2] - (u_sol_x[:, :-1] + u_sol_x[:, 1:]) / 2.0)**2).sum()
        + 1.0/36.0 * ((u_ana_x2[:-1:2, 2::2] - u_sol_x[:, 1:])**2).sum()
        + 1.0/9.0 * ((u_ana_x2[1::2, :-1:2] - u_sol_x[:, :-1])**2).sum()
        + 4.0/9.0 * ((u_ana_x2[1::2, 1::2] - (u_sol_x[:, :-1] + u_sol_x[:, 1:]) / 2.0)**2).sum()
        + 1.0/9.0 * ((u_ana_x2[1::2, 2::2] - u_sol_x[:, 1:])**2).sum()
        + 1.0/36.0 * ((u_ana_x2[2::2, :-1:2] - u_sol_x[:, :-1])**2).sum()
        + 1.0/9.0 * ((u_ana_x2[2::2, 1::2] - (u_sol_x[:, :-1] + u_sol_x[:, 1:]) / 2.0)**2).sum()
        + 1.0/36.0 * ((u_ana_x2[2::2, 2::2] - u_sol_x[:, 1:])**2).sum()
        + 1.0/36.0 * ((u_ana_y2[:-1:2, :-1:2] - u_sol_y[:-1, :])**2).sum()
        + 1.0/9.0 * ((u_ana_y2[:-1:2, 1::2] - u_sol_y[:-1, :])**2).sum()
        + 1.0/36.0 * ((u_ana_y2[:-1:2, 2::2] - u_sol_y[:-1, :])**2).sum()
        + 1.0/9.0 * ((u_ana_y2[1::2, :-1:2] - (u_sol_y[:-1, :] + u_sol_y[1:, :]) / 2.0)**2).sum()
        + 4.0/9.0 * ((u_ana_y2[1::2, 1::2] - (u_sol_y[:-1, :] + u_sol_y[1:, :]) / 2.0)**2).sum()
        + 1.0/9.0 * ((u_ana_y2[1::2, 2::2] - (u_sol_y[:-1, :] + u_sol_y[1:, :]) / 2.0)**2).sum()
        + 1.0/36.0 * ((u_ana_y2[2::2, :-1:2] - u_sol_y[1:, :])**2).sum()
        + 1.0/9.0 * ((u_ana_y2[2::2, 1::2] - u_sol_y[1:, :])**2).sum()
        + 1.0/36.0 * ((u_ana_y2[2::2, 2::2] - u_sol_y[1:, :])**2).sum()
    ))

    return e_linf, e_l2, e_h1
