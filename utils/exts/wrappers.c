#include "exts.h"

static PyObject* solve_cg_2_wrapper(PyObject* self, PyObject* args)
{
    int size, max_it;
    double tol;

    PyObject* data_obj = NULL, * ind_obj = NULL, * ptr_obj = NULL, * vec_obj = NULL, * init_obj = NULL;
    if (!PyArg_ParseTuple(
        args, "iO!O!O!O!O!di",
        &size,
        &PyArray_Type, &data_obj,
        &PyArray_Type, &ind_obj,
        &PyArray_Type, &ptr_obj,
        &PyArray_Type, &vec_obj,
        &PyArray_Type, &init_obj,
        &tol, &max_it
    ))
        return NULL;
    
    PyArrayObject* data_arr = (PyArrayObject*)PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* ind_arr = (PyArrayObject*)PyArray_FROM_OTF(ind_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* ptr_arr = (PyArrayObject*)PyArray_FROM_OTF(ptr_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* vec_arr = (PyArrayObject*)PyArray_FROM_OTF(vec_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* init_arr = (PyArrayObject*)PyArray_FROM_OTF(init_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!data_arr || !ind_arr || !ptr_arr || !vec_arr || !init_arr)
        return NULL;

    double* data = PyArray_DATA(data_arr);
    int* ind = PyArray_DATA(ind_arr);
    int* ptr = PyArray_DATA(ptr_arr);
    double* vec = PyArray_DATA(vec_arr);
    double* init = PyArray_DATA(init_arr);

    sparse_matrix_t mat;
    mkl_sparse_d_create_csr(&mat, SPARSE_INDEX_BASE_ZERO, size, size, ptr, ptr+1, ind, data);
    double* work = malloc(3*size * sizeof(double));

    int ctr = solve_cg_2(size, mat, vec, init, tol, max_it, work);

    free(work);
    mkl_sparse_destroy(mat);

    PyArray_ResolveWritebackIfCopy(init_arr);
    Py_DECREF(init_arr);
    Py_DECREF(vec_arr);
    Py_DECREF(ptr_arr);
    Py_DECREF(ind_arr);
    Py_DECREF(data_arr);
    
    return Py_BuildValue("i", ctr);
}

static PyObject* solve_cg_infty_wrapper(PyObject* self, PyObject* args)
{
    int size, max_it;
    double tol;

    PyObject* data_obj = NULL, * ind_obj = NULL, * ptr_obj = NULL, * vec_obj = NULL, * init_obj = NULL;
    if (!PyArg_ParseTuple(
        args, "iO!O!O!O!O!di",
        &size,
        &PyArray_Type, &data_obj,
        &PyArray_Type, &ind_obj,
        &PyArray_Type, &ptr_obj,
        &PyArray_Type, &vec_obj,
        &PyArray_Type, &init_obj,
        &tol, &max_it
    ))
        return NULL;
    
    PyArrayObject* data_arr = (PyArrayObject*)PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* ind_arr = (PyArrayObject*)PyArray_FROM_OTF(ind_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* ptr_arr = (PyArrayObject*)PyArray_FROM_OTF(ptr_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* vec_arr = (PyArrayObject*)PyArray_FROM_OTF(vec_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* init_arr = (PyArrayObject*)PyArray_FROM_OTF(init_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!data_arr || !ind_arr || !ptr_arr || !vec_arr || !init_arr)
        return NULL;

    double* data = PyArray_DATA(data_arr);
    int* ind = PyArray_DATA(ind_arr);
    int* ptr = PyArray_DATA(ptr_arr);
    double* vec = PyArray_DATA(vec_arr);
    double* init = PyArray_DATA(init_arr);

    sparse_matrix_t mat;
    mkl_sparse_d_create_csr(&mat, SPARSE_INDEX_BASE_ZERO, size, size, ptr, ptr+1, ind, data);
    double* work = malloc(3*size * sizeof(double));

    int ctr = solve_cg_infty(size, mat, vec, init, tol, max_it, work);

    free(work);
    mkl_sparse_destroy(mat);

    PyArray_ResolveWritebackIfCopy(init_arr);
    Py_DECREF(init_arr);
    Py_DECREF(vec_arr);
    Py_DECREF(ptr_arr);
    Py_DECREF(ind_arr);
    Py_DECREF(data_arr);
    
    return Py_BuildValue("i", ctr);
}

static PyMethodDef methods[] = 
{
    {"solve_cg_2_wrapper", solve_cg_2_wrapper, METH_VARARGS, NULL},
    {"solve_cg_infty_wrapper", solve_cg_infty_wrapper, METH_VARARGS, NULL},
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
