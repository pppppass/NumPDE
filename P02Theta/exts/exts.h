#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

int para_theta_ghost_full(int size, int len, double width, double dura, double* coef, double* sour, double* alpha, double* grad, double theta, double* sol, int ldsol, double* work);
int para_theta_direct(int size, int len, double width, double dura, double* coef, double* sour, double* alpha, double* grad, double theta, double* sol, int ldsol, double* work);
int para_theta_ghost_half(int size, int len, double width, double dura, double* coef, double* sour, double* alpha, double* grad, double theta, double* sol, int ldsol, double* work);
