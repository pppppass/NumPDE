import numpy
import scipy.fftpack


def calc_init_1(size):
    n = size
    x = numpy.linspace(1.0 / n, 1.0 - 1.0 / n, n-1)[:, None]
    u = x**2 * (1.0 - x)
    return u


def calc_approx_1(size, time):
    n, t = size, time
    i = numpy.arange(1, n)[:, None]
    c = 1.0 / numpy.pi**3 * (-2.0 - 4.0 * (-1)**i) / i**3
    c_t = numpy.exp(-numpy.pi**2 * t * i**2) * c
    u = scipy.fftpack.idst(c_t.transpose(), type=1).transpose()
    return u


def calc_init_2(size):
    n = size
    x = numpy.linspace(1.0 / n, 1.0 - 1.0 / n, n-1)[:, None]
    f = x < 2.0 / 3.0
    u = f * (2.0 / 9.0 * x) + (~f) * (-4.0 / 9.0 * (x - 1.0))
    return u


def calc_approx_2(size, time):
    n, t = size, time
    i = numpy.arange(1, n)[:, None]
    c = 2.0 / 3.0 / numpy.pi**2 * numpy.sin(2.0 / 3.0 * numpy.pi * i) / i**2
    c_t = numpy.exp(-numpy.pi**2 * t * i**2) * c
    u = scipy.fftpack.idst(c_t.transpose(), type=1).transpose()
    return u


def calc_init_3(size):
    n = size
    x = numpy.linspace(1.0 / n, 1.0 - 1.0 / n, n-1)[:, None]
    f = (x > 1.0 / 3.0 - 1.0e-15) & (x < 5.0 / 6.0 + 1.0e-15)
    u = f * (4.0 / 27.0)
    return u


def calc_approx_3(size, time):
    n, t = size, time
    i = numpy.arange(1, n)[:, None]
    c = -4.0 / 27.0 / numpy.pi * (numpy.cos(5.0 / 6.0 * numpy.pi * i) - numpy.cos(numpy.pi / 3.0 * i)) / i
    c_t = numpy.exp(-numpy.pi**2 * t * i**2) * c
    u = scipy.fftpack.idst(c_t.transpose(), type=1).transpose()
    return u
