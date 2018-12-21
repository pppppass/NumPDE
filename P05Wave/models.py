import numpy


def calc_func_1(para):
    p = para
    u = p**2 * (1.0 - p)
    return u


def calc_func_2(para):
    p = para
    f_1 = p >= 1.0 / 3.0
    f_2 = p >= 2.0 / 3.0
    u = 4.0 / 27.0 * ((f_1 & ~f_2) * (-1.0 + 3.0 * p) + (f_1 & f_2) * (3.0 - 3.0 * p))
    return u


def calc_func_3(para):
    p = para
    u = 2.0/27.0 * (1.0 - numpy.cos(2.0 * numpy.pi * p))
    return u


def calc_func_4(para):
    p = para
    u = 1.0/27.0 * (1.0 - numpy.cos(2.0 * numpy.pi * p))**2
    return u


def calc_extend(pos, time, func):
    x, t, f = pos, time, func
    r, s = x - t, x + t
    r_int, s_int = numpy.floor(r + 1.0e-10), numpy.floor(s + 1.0e-10)
    r_fl, s_fl = r - r_int, s - s_int
    r_rem, s_rem = r_int.astype(numpy.int) % 4, s_int.astype(numpy.int) % 4
    u = numpy.zeros_like(r)
    u += (r_rem == 0) * f(r_fl)
    u -= (r_rem == 2) * f(r_fl)
    u -= (s_rem == 1) * f(1.0 - s_fl)
    u += (s_rem == 3) * f(1.0 - s_fl)
    return u


def calc_ana_sol_1(pos, time):
    x, t = pos, time
    u = calc_extend(x, t, calc_func_1)
    return u


def calc_ana_sol_2(pos, time):
    x, t = pos, time
    u = calc_extend(x, t, calc_func_2)
    return u


def calc_ana_sol_3(pos, time):
    x, t = pos, time
    u = calc_extend(x, t, calc_func_3)
    return u


def calc_ana_sol_4(pos, time):
    x, t = pos, time
    u = calc_extend(x, t, calc_func_4)
    return u


def calc_ana_sol_t_init_1(pos):
    x = pos
    v = 3.0 * x**2 - 2.0 * x
    return v


def calc_ana_sol_t_init_2(pos):
    x = pos
    v = numpy.zeros_like(x)
    f_1 = x >= 1.0 / 3.0
    f_2 = x >= 2.0 / 3.0
    v[f_1 & ~f_2] = -4.0 / 9.0
    v[f_1 & f_2] = 4.0 / 9.0
    return v


def calc_ana_sol_t_init_3(pos):
    x = pos
    v = -4.0/27.0 * numpy.pi * numpy.sin(2.0 * numpy.pi * x)
    return v


def calc_ana_sol_t_init_4(pos):
    x = pos
    v = -4.0/27.0 * numpy.pi * numpy.sin(2.0 * numpy.pi * x) * (1.0 - numpy.cos(2.0 * numpy.pi * x))
    return v


def calc_coef(x, t):
    a = numpy.ones(numpy.broadcast(x, t).shape)
    return a


def calc_alpha(t):
    alpha_1 = numpy.zeros_like(t)[0, :]
    alpha_2 = numpy.ones_like(t)[0, :]
    return numpy.vstack([alpha_1, alpha_2])


def calc_grad(t):
    g_1 = numpy.zeros_like(t)[0, :]
    g_2 = numpy.zeros_like(t)[0, :]
    return numpy.vstack([g_1, g_2])


def get_dirich_flag():
    return numpy.array([False, True])
