import numpy


def get_beta_1():
    beta = numpy.array([0.0, 1.0, 2.0])
    return beta


def get_beta_2():
    beta = numpy.array([2.0, 0.0, 1.0])
    return beta


def get_cond_1():
    g1 = lambda y: numpy.zeros_like(y)
    g2 = lambda x: numpy.zeros_like(x)
    g3 = lambda y: -numpy.pi * numpy.cos(numpy.pi * y)
    g4 = lambda x: -2.0 * numpy.sin(numpy.pi * x)
    f = lambda x, y: 2.0 * numpy.pi**2 * numpy.sin(numpy.pi * x) * numpy.cos(numpy.pi * y)
    return g1, g2, g3, g4, f


def get_cond_2():
    g1 = lambda y: numpy.log(1.0 + y**2) / 2.0 + numpy.arctan(y)
    g2 = lambda x: 2.0 * numpy.log(x + 1.0) - 1.0 / (x + 1.0)
    g3 = lambda y: (2.0 - y) / (4.0 + y**2)
    g4 = lambda x: numpy.log(1.0 + (x + 1.0)**2) / 2.0 + numpy.pi / 2.0 - numpy.arctan(x + 1.0) + (x + 2.0) / ((x + 1.0)**2 + 1.0)
    f = lambda x, y: numpy.zeros_like(numpy.broadcast_arrays(x, y)[0])
    return g1, g2, g3, g4, f


get_ana_1 = lambda x, y: numpy.sin(numpy.pi * x) * numpy.cos(numpy.pi * y)
get_ana_x_1 = lambda x, y: numpy.pi * numpy.cos(numpy.pi * x) * numpy.cos(numpy.pi * y)
get_ana_y_1 = lambda x, y: -numpy.pi * numpy.sin(numpy.pi * x) * numpy.sin(numpy.pi * y)


get_ana_2 = lambda x, y: numpy.log((x + 1.0)**2 + y**2) / 2.0 + numpy.arctan(y / (x + 1.0))
get_ana_x_2 = lambda x, y: (x + 1.0 - y) / ((x + 1.0)**2 + y**2)
get_ana_y_2 = lambda x, y: (x + 1.0 + y) / ((x + 1.0)**2 + y**2)


def driver_ana_1(size):
    n = size
    t = numpy.linspace(0.0, 1.0, n+1)
    x, y = t[:, None], t[None, :]
    u_ana = get_ana_1(x, y)
    return u_ana


def driver_ana_2(size):
    n = size
    t = numpy.linspace(0.0, 1.0, n+1)
    x, y = t[:, None], t[None, :]
    u_ana = get_ana_2(x, y)
    return u_ana


def driver_ana2_all_1(size):
    n = size
    t2 = numpy.linspace(0.0, 1.0, 2*n+1)
    x2, y2 = t2[:, None], t2[None, :]
    u_ana2 = get_ana_1(x2, y2)
    u_ana_x2 = get_ana_x_1(x2, y2)
    u_ana_y2 = get_ana_y_1(x2, y2)
    return u_ana2, u_ana_x2, u_ana_y2


def driver_ana2_all_2(size):
    n = size
    t2 = numpy.linspace(0.0, 1.0, 2*n+1)
    x2, y2 = t2[:, None], t2[None, :]
    u_ana2 = get_ana_2(x2, y2)
    u_ana_x2 = get_ana_x_2(x2, y2)
    u_ana_y2 = get_ana_y_2(x2, y2)
    return u_ana2, u_ana_x2, u_ana_y2
