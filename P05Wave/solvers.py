import numpy


def calc_first(init, init_t, step, tau, coef, alpha, grad, flag):
    u_0, v_0, h, a, g, f = init, init_t, step, coef, grad, flag
    n = u_0.shape[0] - 1
    nu = tau / h
    u = numpy.zeros(n+1)
    u[1:-1] = (
          nu**2 / 2.0 * a[1:-1, 0]**2 * (u_0[:-2] + u_0[2:])
        + (1.0 - nu**2 * a[1:-1, 0]**2) * u_0[1:-1]
        + tau * v_0[1:-1]
    )
    if f[0]:
        u[0] = g[0, 1]
    else:
        u[0] = (
              nu**2 * a[0, 0]**2 * u_0[1]
            + (1.0 - nu**2 * (1.0 + h * alpha[0, 0]) * a[0, 0]**2) * u_0[0]
            + tau * v_0[0]
            + h * nu**2 * a[0, 0]**2 * g[0, 0]
        )
    if f[1]:
        u[-1] = g[1, 1]
    else:
        u[-1] = (
              nu**2 * a[-1, 0]**2 * u_0[-2]
            + (1.0 - nu**2 * (1.0 + h * alpha[1, 0]) * a[-1, 0]**2) * u_0[-1]
            + tau * v_0[-1]
            + h * nu**2 * a[-1, 0]**2 * g[1, 0]
        )
    return u


def iter_expl(value_old, value, step, tau, coef, alpha, grad, flag):
    u_old, u, h, a, g, f = value_old, value, step, coef, grad, flag
    n = u.shape[0] - 1
    nu = tau / h
    u_new = numpy.zeros(n+1)
    u_new[1:-1] = (
          nu**2 * a[1:-1, 0]**2 * (u[:-2] + u[2:])
        + (2.0 - 2.0 * nu**2 * a[1:-1, 0]**2) * u[1:-1]
        - u_old[1:-1]
    )
    if f[0]:
        u_new[0] = g[0, 1]
    else:
        u_new[0] = (
              2.0 * nu**2 * a[0, 0]**2 * u[1]
            + (2.0 - 2.0 * nu**2 * (1.0 + h * alpha[0, 0]) * a[0, 0]**2) * u[0]
            - u_old[0]
            + 2.0 * h * nu**2 * a[0, 0]**2 * g[0, 0]
        )
    if f[1]:
        u_new[-1] = g[1, 1]
    else:
        u_new[-1] = (
              2.0 * nu**2 * a[-1, 0]**2 * u[-2]
            + (2.0 - 2.0 * nu**2 * (1.0 + h * alpha[1, 0]) * a[-1, 0]**2) * u[-1]
            - u_old[-1]
            + 2.0 * h * nu**2 * a[-1, 0]**2 * g[1, 0]
        )
    return u_new


def driver_expl(len_, init, init_t, step, tau, coef, alpha, grad, flag):
    m, u_0, v_0, h, a, g, f = len_, init, init_t, step, coef, grad, flag
    n = u_0.shape[0] - 1
    u = numpy.zeros((n+1, m+1))
    u[:, 0] = u_0[:, 0]
    u[:, 1] = calc_first(u_0[:, 0], v_0[:, 0], h, tau, a[:, :2], alpha[:, :2], g[:, :2], f)
    for i in range(1, m):
        u[:, i+1] = iter_expl(u[:, i-1], u[:, i], h, tau, a[:, i:i+2], alpha[:, i:i+2], g[:, i:i+2], f)
    return u


# def driver_expl(len_, init, init_t, step, tau, coef, alpha, grad, flag):
#     m, u_0, v_0, h, a, g, f = len_, init, init_t, step, coef, grad, flag
#     u_old = u_0[:, 0]
#     u = calc_first(u_0[:, 0], v_0[:, 0], h, tau, a[:, :2], alpha[:, :2], g[:, :2], f)
#     for i in range(1, m):
#         u_new = iter_expl(u_old, u, h, tau, a[:, i:i+2], alpha[:, i:i+2], g[:, i:i+2], f)
#         u_old = u
#         u = u_new
#     return u
