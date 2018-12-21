import numpy


def get_riemann_ana(size, time, width=5.0):
    n, t, w = size, time, width
    n_total = int(2.0 * w * n + 1.0e-5)
    n_jump = int((w + time / 2.0) * n + 0.5)
    x = numpy.linspace(-w + 0.5 / n, w - 0.5 / n, n_total)
    u = numpy.zeros((n_total))
    u[:n_jump] = 1.0
    return x, u


def get_sin_ana(size, time, width=5.0, iters=20):
    n, t, w = size, time, width
    n_total = int(2.0 * w * n + 1.0e-5)
    n_left = int((w + -numpy.pi / 2.0 + t) * n + 0.5)
    n_middle = int((w + t / 2.0) * n + 0.5)
    n_right = int((w + numpy.pi / 2.0) * n + 0.5)
    x = numpy.linspace(-w + 0.5 / n, w - 0.5 / n, n_total)
    u = numpy.zeros((n_total))
    u[:n_left], u[n_right:] = 1.0, 0.0
    if t > numpy.pi:
        u[:n_middle] = 1.0
    else:
        u[:n_left], u[n_right:] = 1.0, 0.0
        y = -numpy.pi / 2.0 * numpy.ones((n_middle - n_left))
        for i in range(iters):
            y -= (y + t / 2.0 * (1.0 - numpy.sin(y)) - x[n_left:n_middle]) / (1.0 - t / 2.0 * numpy.cos(y))
        u[n_left:n_middle] = (1.0 - numpy.sin(y)) / 2.0
        y = numpy.pi / 2.0 * numpy.ones((n_right - n_middle))
        for i in range(iters):
            y -= (y + t / 2.0 * (1.0 - numpy.sin(y)) - x[n_middle:n_right]) / (1.0 - t / 2.0 * numpy.cos(y))
        u[n_middle:n_right]= (1.0 - numpy.sin(y)) / 2.0
    return x, u
