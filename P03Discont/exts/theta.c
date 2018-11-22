#include "exts.h"

int para_theta_model(int size, int len, double dura, double theta, double* sol, double* work)
{
    int n = size, m = len;
    double h = 1.0 / n, tau = dura / m;
    double* u = sol, * cd_old = work, * cl_old = work+(n-1), * cd_new = work+2*(n-1), *cl_new = work+3*(n-1), * t = work+4*(n-1);

    double mu = tau / h / h;
    
    for (int j = 0; j < n-1; j++)
    {
        cd_old[j] = 1.0 - 2.0 * mu * (1.0 - theta);
        cl_old[j] = mu * (1.0 - theta);
    }

    int i = 0;
    for (i = 0; i < m; i++)
    {
        for (int j = 0; j < n-1; j++)
            t[j] = 0.0;
        
        {
            int n_ = n-1, nrhs = 1, ldx = 1, ldb = 1;
            double alpha = 1.0, beta = 1.0;
            dlagtm("N", &n_, &nrhs, &alpha, cl_old+1, cd_old, cl_old, u, &ldx, &beta, t, &ldb);
        }
        cblas_dcopy(n-1, t, 1, u, 1);
        
        for (int j = 0; j < n-1; j++)
        {
            cd_new[j] = 1.0 + 2.0 * mu * theta;
            cl_new[j] = -mu * theta;
            t[j] = -mu * theta;
        }
        LAPACKE_dgtsv(LAPACK_ROW_MAJOR, n-1, 1, cl_new+1, cd_new, t, u, 1);
    }

    return i;
}
