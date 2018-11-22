#include "exts.h"

static PyObject* para_theta_ghost_full_wrapper(PyObject* self, PyObject* args)
{
    int size, len, ldsol;
    double width, dura, theta;

    PyObject* coef_obj = NULL, * sour_obj = NULL, * alpha_obj = NULL, * grad_obj = NULL, * sol_obj = NULL;
    if (!PyArg_ParseTuple(
        args, "iiddO!O!O!O!dO!i",
        &size,
        &len,
        &width,
        &dura,
        &PyArray_Type, &coef_obj,
        &PyArray_Type, &sour_obj,
        &PyArray_Type, &alpha_obj,
        &PyArray_Type, &grad_obj,
        &theta,
        &PyArray_Type, &sol_obj,
        &ldsol
    ))
        return NULL;
    
    PyArrayObject* coef_arr = (PyArrayObject*)PyArray_FROM_OTF(coef_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* sour_arr = (PyArrayObject*)PyArray_FROM_OTF(sour_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* alpha_arr = (PyArrayObject*)PyArray_FROM_OTF(alpha_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* grad_arr = (PyArrayObject*)PyArray_FROM_OTF(grad_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* sol_arr = (PyArrayObject*)PyArray_FROM_OTF(sol_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!coef_arr || !sour_arr || !alpha_arr || !grad_arr || !sol_arr)
        return NULL;

    double* coef = PyArray_DATA(coef_arr);
    double* sour = PyArray_DATA(sour_arr);
    double* alpha = PyArray_DATA(alpha_arr);
    double* grad = PyArray_DATA(grad_arr);
    double* sol = PyArray_DATA(sol_arr);

    double* work = malloc(4*(size+1) * sizeof(double));

    int ctr = para_theta_ghost_full(size, len, width, dura, coef, sour, alpha, grad, theta, sol, ldsol, work);

    free(work);

    PyArray_ResolveWritebackIfCopy(sol_arr);
    Py_DECREF(sol_arr);
    Py_DECREF(grad_arr);
    Py_DECREF(alpha_arr);
    Py_DECREF(sour_arr);
    Py_DECREF(coef_arr);
    
    return Py_BuildValue("i", ctr);
}

static PyObject* para_theta_direct_wrapper(PyObject* self, PyObject* args)
{
    int size, len, ldsol;
    double width, dura, theta;

    PyObject* coef_obj = NULL, * sour_obj = NULL, * alpha_obj = NULL, * grad_obj = NULL, * sol_obj = NULL;
    if (!PyArg_ParseTuple(
        args, "iiddO!O!O!O!dO!i",
        &size,
        &len,
        &width,
        &dura,
        &PyArray_Type, &coef_obj,
        &PyArray_Type, &sour_obj,
        &PyArray_Type, &alpha_obj,
        &PyArray_Type, &grad_obj,
        &theta,
        &PyArray_Type, &sol_obj,
        &ldsol
    ))
        return NULL;
    
    PyArrayObject* coef_arr = (PyArrayObject*)PyArray_FROM_OTF(coef_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* sour_arr = (PyArrayObject*)PyArray_FROM_OTF(sour_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* alpha_arr = (PyArrayObject*)PyArray_FROM_OTF(alpha_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* grad_arr = (PyArrayObject*)PyArray_FROM_OTF(grad_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* sol_arr = (PyArrayObject*)PyArray_FROM_OTF(sol_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!coef_arr || !sour_arr || !alpha_arr || !grad_arr || !sol_arr)
        return NULL;

    double* coef = PyArray_DATA(coef_arr);
    double* sour = PyArray_DATA(sour_arr);
    double* alpha = PyArray_DATA(alpha_arr);
    double* grad = PyArray_DATA(grad_arr);
    double* sol = PyArray_DATA(sol_arr);

    double* work = malloc((2*(len+1) + 4*(size+1)) * sizeof(double));

    int ctr = para_theta_direct(size, len, width, dura, coef, sour, alpha, grad, theta, sol, ldsol, work);

    free(work);

    PyArray_ResolveWritebackIfCopy(sol_arr);
    Py_DECREF(sol_arr);
    Py_DECREF(grad_arr);
    Py_DECREF(alpha_arr);
    Py_DECREF(sour_arr);
    Py_DECREF(coef_arr);
    
    return Py_BuildValue("i", ctr);
}

static PyObject* para_theta_ghost_half_wrapper(PyObject* self, PyObject* args)
{
    int size, len, ldsol;
    double width, dura, theta;

    PyObject* coef_obj = NULL, * sour_obj = NULL, * alpha_obj = NULL, * grad_obj = NULL, * sol_obj = NULL;
    if (!PyArg_ParseTuple(
        args, "iiddO!O!O!O!dO!i",
        &size,
        &len,
        &width,
        &dura,
        &PyArray_Type, &coef_obj,
        &PyArray_Type, &sour_obj,
        &PyArray_Type, &alpha_obj,
        &PyArray_Type, &grad_obj,
        &theta,
        &PyArray_Type, &sol_obj,
        &ldsol
    ))
        return NULL;
    
    PyArrayObject* coef_arr = (PyArrayObject*)PyArray_FROM_OTF(coef_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* sour_arr = (PyArrayObject*)PyArray_FROM_OTF(sour_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* alpha_arr = (PyArrayObject*)PyArray_FROM_OTF(alpha_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* grad_arr = (PyArrayObject*)PyArray_FROM_OTF(grad_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* sol_arr = (PyArrayObject*)PyArray_FROM_OTF(sol_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!coef_arr || !sour_arr || !alpha_arr || !grad_arr || !sol_arr)
        return NULL;

    double* coef = PyArray_DATA(coef_arr);
    double* sour = PyArray_DATA(sour_arr);
    double* alpha = PyArray_DATA(alpha_arr);
    double* grad = PyArray_DATA(grad_arr);
    double* sol = PyArray_DATA(sol_arr);

    double* work = malloc((2*(len+1) + 4*size) * sizeof(double));

    int ctr = para_theta_ghost_half(size, len, width, dura, coef, sour, alpha, grad, theta, sol, ldsol, work);

    free(work);

    PyArray_ResolveWritebackIfCopy(sol_arr);
    Py_DECREF(sol_arr);
    Py_DECREF(grad_arr);
    Py_DECREF(alpha_arr);
    Py_DECREF(sour_arr);
    Py_DECREF(coef_arr);
    
    return Py_BuildValue("i", ctr);
}

static PyMethodDef methods[] = 
{
    {"para_theta_ghost_full_wrapper", para_theta_ghost_full_wrapper, METH_VARARGS, NULL},
    {"para_theta_direct_wrapper", para_theta_direct_wrapper, METH_VARARGS, NULL},
    {"para_theta_ghost_half_wrapper", para_theta_ghost_half_wrapper, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef table = 
{
    PyModuleDef_HEAD_INIT,
    "exts", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_exts(void)
{
    import_array();
    return PyModule_Create(&table);
}

