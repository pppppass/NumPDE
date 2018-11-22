import numpy


def calc_ana_sol_1(x, t):
    u = numpy.broadcast_arrays(numpy.cos(numpy.pi * x), t)[0]
    return u


def calc_coef_1(x, t):
    a = numpy.ones(numpy.broadcast(x, t).shape)
    return a


def calc_sour_1(x, t):
    f = numpy.broadcast_arrays(numpy.pi**2 * numpy.cos(numpy.pi * x) , t)[0]
    return f


def calc_alpha_1(t):
    alpha_1 = numpy.zeros_like(t)[0, :]
    alpha_2 = numpy.zeros_like(t)[0, :]
    return numpy.hstack([alpha_1, alpha_2])


def calc_grad_1(t):
    g_1 = numpy.zeros_like(t)[0, :]
    g_2 = numpy.zeros_like(t)[0, :]
    return numpy.hstack([g_1, g_2])


def calc_ana_sol_2(x, t):
    u = numpy.exp(-((t + 1.0) * numpy.log(t + 1.0) - t) / 10.0) * numpy.cos(x - 1.0)
    return u


def calc_coef_2(x, t):
    a = numpy.broadcast_arrays(numpy.log(t + 1.0) / 10.0, x)[0]
    return a


def calc_sour_2(x, t):
    f = numpy.zeros(numpy.broadcast(x, t).shape)
    return f


def calc_alpha_2(t):
    alpha_1 = (numpy.tan(1.0) * numpy.ones_like(t))[0, :]
    alpha_2 = numpy.zeros_like(t)[0, :]
    return numpy.hstack([alpha_1, alpha_2])


def calc_grad_2(t):
    g_1 = numpy.zeros_like(t)[0, :]
    g_2 = numpy.zeros_like(t)[0, :]
    return numpy.hstack([g_1, g_2])


def calc_ana_sol_3(x, t):
    u = (
          1.0 / numpy.sqrt(2.0 * t + numpy.cos(t))
        * numpy.exp(-numpy.tan(x)**2 / (4.0 * (2.0 * t + numpy.cos(t))))
    )
    return u


def calc_coef_3(x, t):
    a = (2.0 - numpy.sin(t)) * numpy.cos(x)**4
    return a


def calc_sour_3(x, t):
    f = (
          (2.0 - numpy.sin(t))
        / numpy.sqrt(2.0 * t + numpy.cos(t))**3
        * numpy.exp(-numpy.tan(x)**2 / (4.0 * (2.0 * t + numpy.cos(t))))
        * numpy.sin(x)**2
    )
    return f


def calc_alpha_3(t):
    alpha_1 = numpy.zeros_like(t)[0, :]
    alpha_2 = ((numpy.tan(1.0) / numpy.cos(1.0)**2 / (2.0 * t + numpy.cos(t))))[0, :]
    return numpy.hstack([alpha_1, alpha_2])


def calc_grad_3(t):
    g_1 = numpy.zeros_like(t)[0, :]
    g_2 = (
          numpy.tan(1.0) / 2.0 / numpy.cos(1.0)**2
        / numpy.sqrt(2.0 * t + numpy.cos(t))**3
        * numpy.exp(-numpy.tan(1.0)**2 / (4.0 * (2.0 * t + numpy.cos(t))))
    )[0, :]
    return numpy.hstack([g_1, g_2])
