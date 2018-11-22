#include "exts.h"

int solve_cg_2(int size, sparse_matrix_t mat, double* vec, double* init, double tol, int max_it, double* work)
{
    int n = size, m = max_it;
    double alpha, beta, rho, rho_, eps = tol;
    double* b = vec, * x = init, * r = work, * p = work+n, * w = work+2*n;
    sparse_matrix_t a = mat;
    
    cblas_dcopy(n, b, 1, r, 1);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, a, (struct matrix_descr){SPARSE_MATRIX_TYPE_GENERAL, 0, 0}, x, 1.0, r);
    rho = cblas_ddot(n, r, 1, r, 1);

    int k = 0;
    while (rho > eps*eps && k < m)
    {
        k++;

        if (k == 1)
            cblas_dcopy(n, r, 1, p, 1);
        else
        {
            beta = rho / rho_;
            cblas_dscal(n, beta, p, 1);
            cblas_daxpy(n, 1.0, r, 1, p, 1);
        }
        
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, a, (struct matrix_descr){SPARSE_MATRIX_TYPE_GENERAL, 0, 0}, p, 0.0, w);
        alpha = rho / cblas_ddot(n, p, 1, w, 1);
        cblas_daxpy(n, alpha, p, 1, x, 1);
        cblas_daxpy(n, -alpha, w, 1, r, 1);
        rho_ = rho;
        rho = cblas_ddot(n, r, 1, r, 1);
    }

    return k;
}

int solve_cg_infty(int size, sparse_matrix_t mat, double* vec, double* init, double tol, int max_it, double* work)
{
    int n = size, m = max_it;
    double alpha, beta, rho, rho_, eps = tol, mu;
    double* b = vec, * x = init, * r = work, * p = work+n, * w = work+2*n;
    sparse_matrix_t a = mat;
    
    cblas_dcopy(n, b, 1, r, 1);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, a, (struct matrix_descr){SPARSE_MATRIX_TYPE_GENERAL, 0, 0}, x, 1.0, r);
    mu = fabs(r[cblas_idamax(n, r, 1)]);
    rho = cblas_ddot(n, r, 1, r, 1);

    int ctr = 0;
    while (mu > eps && ctr < m)
    {
        ctr++;

        if (ctr == 1)
            cblas_dcopy(n, r, 1, p, 1);
        else
        {
            beta = rho / rho_;
            cblas_dscal(n, beta, p, 1);
            cblas_daxpy(n, 1.0, r, 1, p, 1);
        }
        
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, a, (struct matrix_descr){SPARSE_MATRIX_TYPE_GENERAL, 0, 0}, p, 0.0, w);
        alpha = rho / cblas_ddot(n, p, 1, w, 1);
        cblas_daxpy(n, alpha, p, 1, x, 1);
        cblas_daxpy(n, -alpha, w, 1, r, 1);
        rho_ = rho;
        rho = cblas_ddot(n, r, 1, r, 1);
        mu = fabs(r[cblas_idamax(n, r, 1)]);
    }

    return ctr;
}
