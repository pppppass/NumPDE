#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

int solve_cg_2(int size, sparse_matrix_t mat, double* vec, double* init, double tol, int max_it, double* work);
int solve_cg_infty(int size, sparse_matrix_t mat, double* vec, double* init, double tol, int max_it, double* work);
