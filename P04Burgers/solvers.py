import numpy


def iter_upwind_non(value, step, tau):
    u, h = value, step
    s = u >= 0
    u_aug = numpy.hstack([u[0], u, u[-1]])
    g = u_aug[1:] - u_aug[:-1]
    u -= tau / h * u * (s * g[:-1] + (~s) * g[1:])
    return u


def iter_upwind_con(value, step, tau):
    u, h = value, step
    u_aug = numpy.hstack([u[0], u, u[-1]])
    f_u = u_aug**2 / 2.0
    a = u_aug[:-1] + u_aug[1:]
    s = a >= 0
    f = s * (s * f_u[:-1] + (~s) * f_u[1:])
    u -= tau / h * (f[1:] - f[:-1])


def iter_rm(value, step, tau):
    u, h = value, step
    u_aug = numpy.hstack([u[0], u, u[-1]])
    f_u = u_aug**2 / 2.0
    u_mid = (u_aug[:-1] + u_aug[1:]) / 2.0 - tau / 2.0 / h * (f_u[1:] - f_u[:-1])
    u -= tau / 2.0 / h * (u_mid[1:]**2 - u_mid[:-1]**2)


def iter_lw(value, step, tau):
    u, h = value, step
    u_aug = numpy.hstack([u[0], u, u[-1]])
    f_u = u_aug**2 / 2.0
    a = (u_aug[1:] + u_aug[:-1]) / 2.0
    u += -tau / 2.0 / h * (f_u[2:] - f_u[:-2]) + tau**2 / 2.0 / h**2 * (a[1:] * (f_u[2:] - f_u[1:-1]) - a[:-1] * (f_u[1:-1] - f_u[:-2]))
