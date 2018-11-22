#include "exts.h"

int para_theta_ghost_full(int size, int len, double width, double dura, double* coef, double* sour, double* alpha, double* grad, double theta, double* sol, int ldsol, double* work)
{
    int n = size, m = len, m_ = ldsol;
    double h = width / n, tau = dura / m;
    double* a = coef, * f = sour, * alpha1 = alpha, * alpha2 = alpha+(m+1), * g1 = grad, * g2 = grad+(m+1), * u_sol = sol, * cd = work, * cl = work+(n+1), * u = work+2*(n+1), * t = work+3*(n+1);

    double (*a_)[m+1] = a, (*f_)[m+1] = f;

    double mu = tau / h / h;

    cblas_dcopy(n+1, u_sol, m_, u, 1);

    int i = 0;
    for (i = 0; i < m; i++)
    {

        for (int j = 0; j < n+1; j++) t[j] = 0.0;
        cblas_daxpy(n+1, tau * (1.0 - theta), f+i, m+1, t, 1);
        cblas_daxpy(n+1, tau * theta, f+(i+1), m+1, t, 1);
        t[0] += 2.0 * mu * h * ((1.0 - theta) * g1[i] * a_[0][i] + theta * g1[i+1] * a_[0][i+1]);
        t[n] += 2.0 * mu * h * ((1.0 - theta) * g2[i] * a_[n][i] + theta * g2[i+1] * a_[n][i+1]);
        
        for (int j = 0; j < n+1; j++) cd[j] = 1.0;
        cblas_daxpy(n+1, -2.0 * mu * (1.0 - theta), a+i, m+1, cd, 1);
        cd[0] -= 2.0 * mu * h * (1.0 - theta) * alpha1[i] * a_[0][i];
        cd[n] -= 2.0 * mu * h * (1.0 - theta) * alpha2[i] * a_[n][i];
        for (int j = 0; j < n+1; j++) cl[j] = 0.0;
        cblas_daxpy(n+1, mu * (1.0 - theta), a+i, m+1, cl, 1);
        cl[0] *= 2.0;
        cl[n] *= 2.0;
        {
            int n_ = n+1, nrhs = 1, ldx = 1, ldb = 1;
            double alpha = 1.0, beta = 1.0;
            dlagtm("N", &n_, &nrhs, &alpha, cl+1, cd, cl, u, &ldx, &beta, t, &ldb);
        }
        cblas_dcopy(n+1, t, 1, u, 1);
        
        for (int j = 0; j < n+1; j++) cd[j] = 1.0;
        cblas_daxpy(n+1, 2.0 * mu * theta, a+(i+1), m+1, cd, 1);
        cd[0] += 2.0 * mu * h * theta * alpha1[i+1] * a_[0][i+1];
        cd[n] += 2.0 * mu * h * theta * alpha2[i+1] * a_[n][i+1];
        for (int j = 0; j < n+1; j++) cl[j] = 0.0;
        cblas_daxpy(n+1, -mu * theta, a+(i+1), m+1, cl, 1);
        cl[0] *= 2.0;
        cl[n] *= 2.0;
        cblas_dcopy(n+1, cl, 1, t, 1);
        LAPACKE_dgtsv(LAPACK_ROW_MAJOR, n+1, 1, cl+1, cd, t, u, 1);
        
        cblas_dcopy(n+1, u, 1, u_sol+(i+1), m_);
    }

    return i;
}

int para_theta_direct(int size, int len, double width, double dura, double* coef, double* sour, double* alpha, double* grad, double theta, double* sol, int ldsol, double* work)
{
    int n = size, m = len, m_ = ldsol;
    double h = width / n, tau = dura / m;
    double* a = coef, * f = sour, * alpha1 = alpha, * alpha2 = alpha+(m+1), * g1 = grad, * g2 = grad+(m+1), * u_sol = sol, * beta1 = work, * beta2 = work+(m+1), * cd = work+2*(m+1), * cl = work+2*(m+1)+(n+1), * u = work+2*(m+1)+2*(n+1), * t = work+2*(m+1)+3*(n+1);

    double (*a_)[m+1] = a, (*f_)[m+1] = f;

    double mu = tau / h / h;

    vdLinearFrac(m+1, alpha1, alpha1, 0.0, 1.0, h, 1.0, beta1);
    vdLinearFrac(m+1, alpha2, alpha2, 0.0, 1.0, h, 1.0, beta2);

    cblas_dcopy(n+1, u_sol, m_, u, 1);

    int i = 0;
    for (i = 0; i < m; i++)
    {

        for (int j = 1; j < n; j++) t[j] = 0.0;
        cblas_daxpy(n-1, tau * (1.0 - theta), f+i+(m+1), m+1, t+1, 1);
        cblas_daxpy(n-1, tau * theta, f+(i+1)+(m+1), m+1, t+1, 1);
        t[1] += mu * h * ((1.0 - theta) * beta1[i] * g1[i] * a_[1][i] + theta * beta1[i+1] * g1[i+1] * a_[1][i+1]);
        t[n-1] += mu * h * ((1.0 - theta) * beta2[i] * g2[i] * a_[n-1][i] + theta * beta2[i+1] * g2[i+1] * a_[n-1][i+1]);
        
        for (int j = 1; j < n; j++) cd[j] = 1.0;
        cblas_daxpy(n-1, -2.0 * mu * (1.0 - theta), a+i+(m+1), m+1, cd+1, 1);
        cd[1] += mu * (1.0 - theta) * beta1[i] * a_[1][i];
        cd[n-1] += mu * (1.0 - theta) * beta2[i] * a_[n-1][i];
        for (int j = 1; j < n; j++) cl[j] = 0.0;
        cblas_daxpy(n-1, mu * (1.0 - theta), a+i+(m+1), m+1, cl+1, 1);
        {
            int n_ = n-1, nrhs = 1, ldx = 1, ldb = 1;
            double alpha = 1.0, beta = 1.0;
            dlagtm("N", &n_, &nrhs, &alpha, cl+2, cd+1, cl+1, u+1, &ldx, &beta, t+1, &ldb);
        }
        cblas_dcopy(n-1, t+1, 1, u+1, 1);
        
        for (int j = 1; j < n; j++) cd[j] = 1.0;
        cblas_daxpy(n-1, 2.0 * mu * theta, a+(i+1)+(m+1), m+1, cd+1, 1);
        cd[1] -= mu * theta * beta1[i+1] * a_[1][i+1];
        cd[n-1] -= mu * theta * beta2[i+1] * a_[n-1][i+1];
        for (int j = 1; j < n; j++) cl[j] = 0.0;
        cblas_daxpy(n-1, -mu * theta, a+(i+1)+(m+1), m+1, cl+1, 1);
        cblas_dcopy(n-1, cl+1, 1, t+1, 1);
        LAPACKE_dgtsv(LAPACK_ROW_MAJOR, n-1, 1, cl+2, cd+1, t+1, u+1, 1);
        
        u[0] = beta1[i+1] * u[1] + h * beta1[i+1] * g1[i];
        u[n] = beta2[i+1] * u[n-1] + h * beta2[i+1] * g2[i];
        
        cblas_dcopy(n+1, u, 1, u_sol+(i+1), m_);
    }

    return i;
}

int para_theta_ghost_half(int size, int len, double width, double dura, double* coef, double* sour, double* alpha, double* grad, double theta, double* sol, int ldsol, double* work)
{
    int n = size, m = len, m_ = ldsol;
    double h = width / n, tau = dura / m;
    double* a = coef, * f = sour, * alpha1 = alpha, * alpha2 = alpha+(m+1), * g1 = grad, * g2 = grad+(m+1), * u_sol = sol, * xi1 = work, * xi2 = work+(m+1), * cd = work+2*(m+1), * cl = work+2*(m+1)+n, * u = work+2*(m+1)+2*n, * t = work+2*(m+1)+3*n;

    double (*a_)[m+1] = a, (*f_)[m+1] = f;

    double mu = tau / h / h;

    vdLinearFrac(m+1, alpha1, alpha1, -h, 2.0, h, 2.0, xi1);
    vdLinearFrac(m+1, alpha2, alpha2, -h, 2.0, h, 2.0, xi2);

    cblas_dcopy(n, u_sol, m_, u, 1);

    int i = 0;
    for (i = 0; i < m; i++)
    {

        for (int j = 0; j < n; j++) t[j] = 0.0;
        cblas_daxpy(n, tau * (1.0 - theta), f+i, m+1, t, 1);
        cblas_daxpy(n, tau * theta, f+(i+1), m+1, t, 1);
        t[0] += mu * h * ((1.0 - theta) * (1.0 + xi1[i]) / 2.0 * g1[i] * a_[0][i] + theta * (1.0 + xi1[i+1]) / 2.0 * g1[i+1] * a_[0][i+1]);
        t[n-1] += mu * h * ((1.0 - theta) * (1.0 + xi2[i]) / 2.0 * g2[i] * a_[n-1][i] + theta * (1.0 + xi2[i+1]) / 2.0 * g2[i+1] * a_[n-1][i+1]);
        
        for (int j = 0; j < n; j++) cd[j] = 1.0;
        cblas_daxpy(n, -2.0 * mu * (1.0 - theta), a+i, m+1, cd, 1);
        cd[0] += mu * (1.0 - theta) * xi1[i] * a_[0][i];
        cd[n-1] += mu * (1.0 - theta) * xi2[i] * a_[n-1][i];
        for (int j = 0; j < n; j++) cl[j] = 0.0;
        cblas_daxpy(n, mu * (1.0 - theta), a+i, m+1, cl, 1);
        {
            int n_ = n, nrhs = 1, ldx = 1, ldb = 1;
            double alpha = 1.0, beta = 1.0;
            dlagtm("N", &n_, &nrhs, &alpha, cl+1, cd, cl, u, &ldx, &beta, t, &ldb);
        }
        cblas_dcopy(n, t, 1, u, 1);
        
        for (int j = 0; j < n; j++) cd[j] = 1.0;
        cblas_daxpy(n, 2.0 * mu * theta, a+(i+1), m+1, cd, 1);
        cd[0] -= mu * theta * xi1[i+1] * a_[0][i+1];
        cd[n-1] -= mu * theta * xi2[i+1] * a_[n-1][i+1];
        for (int j = 0; j < n; j++) cl[j] = 0.0;
        cblas_daxpy(n, -mu * theta, a+(i+1), m+1, cl, 1);
        cblas_dcopy(n, cl, 1, t, 1);
        LAPACKE_dgtsv(LAPACK_ROW_MAJOR, n, 1, cl+1, cd, t, u, 1);
        
        cblas_dcopy(n, u, 1, u_sol+(i+1), m_);
    }

    return i;
}
