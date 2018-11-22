#include "exts.h"

static PyObject* para_theta_model_wrapper(PyObject* self, PyObject* args)
{
    int size, len;
    double dura, theta;

    PyObject* sol_obj = NULL;
    if (!PyArg_ParseTuple(
        args, "iiddO!",
        &size,
        &len,
        &dura,
        &theta,
        &PyArray_Type, &sol_obj
    ))
        return NULL;
    
    PyArrayObject* sol_arr = (PyArrayObject*)PyArray_FROM_OTF(sol_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!sol_arr)
        return NULL;

    double* sol = PyArray_DATA(sol_arr);

    double* work = malloc(5*(size-1) * sizeof(double));

    int ctr = para_theta_model(size, len, dura, theta, sol, work);

    free(work);

    PyArray_ResolveWritebackIfCopy(sol_arr);
    Py_DECREF(sol_arr);
    
    return Py_BuildValue("i", ctr);
}

static PyMethodDef methods[] = 
{
    {"para_theta_model_wrapper", para_theta_model_wrapper, METH_VARARGS, NULL},
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

