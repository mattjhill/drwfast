#include <Python.h>
#include <numpy/arrayobject.h>
#include "tridiagonal.h"

static char module_docstring[] =
    "This module provides an interface for fast calculations of tridiagonal\n"
    "matrices using C.";
static char trimult_docstring[] = 
    "trimult(a, b, c, x)\n\n"
    "Multiply a tridiagonal matrix by a vector.\n\n"
    "Parameters\n"
    "----------\n"
    "a : numpy array\n"
    "   The lower off-diagonal.\n"
    "b :  numpy array\n"
    "   The diagonal.\n"
    "c : numpy array\n"
    "   The upper off-diagonal.\n"
    "x : numpy array\n"
    "   The vector to multply by.\n\n"
    "Returns\n"
    "-------\n"
    "y : numpy array\n"
    "   The product.";

static char trisolve_docstring[] = 
    "trisolve(a, b, c, x)\n\n"
    "Solve a tridiagonal system.\n\n"
    "Parameters\n"
    "----------\n"
    "a : numpy array\n"
    "   The lower off-diagonal.\n"
    "b :  numpy array\n"
    "   The diagonal.\n"
    "c : numpy array\n"
    "   The upper off-diagonal.\n"
    "x : numpy array\n"
    "   The right-hand-side vector.\n\n"
    "Returns\n"
    "-------\n"
    "y : numpy array\n"
    "   The solution.";

static char tridet_docstring[] = 
    "tridet(a, b, c)\n\n"
    "Compute the log determinant of a tridiagonal matrix.\n\n"
    "Parameters\n"
    "----------\n"
    "a : numpy array\n"
    "   The lower off-diagonal.\n"
    "b :  numpy array\n"
    "   The diagonal.\n"
    "c : numpy array\n"
    "   The upper off-diagonal.\n\n"
    "Returns\n"
    "-------\n"
    "logdet : float\n"
    "   The log-determinant.";

static char snsolve_docstring[] = 
    "snsolve(var, tcorr, t, err2, y, ret_det=False)\n\n"
    "Solve a tridiagonal system with noise.\n\n"
    "Parameters\n"
    "----------\n"
    "var : float\n"
    "   The variance of the process.\n"
    "tcorr : float\n"
    "   The relaxation time of the process.\n"
    "t : array_like\n"
    "   The array of times.\n"
    "err2 : array_like\n"
    "   The squared errors.\n"
    "y : array_like\n"
    "   The array of observations.\n"
    "ret_det : boolean, optional\n"
    "   Return the determinant. Default is False.\n\n"
    "Returns\n"
    "-------\n"
    "x : array_like\n"
    "   The solution to system.\n"
    "logdet : float\n"
    "   The determinant if ret_det is True.";

static char lnlike_docstring[] = 
    "lnlike(var, tcorr, t, y, err2)\n\n"
    "calculate the log-likelihood of a process with var and tcorr.\n\n"
    "Parameters\n"
    "----------\n"
    "var : float\n"
    "   The variance of the process.\n"
    "tcorr : float\n"
    "   The relaxation time of the process.\n"
    "t : array_like\n"
    "   The array of times.\n"
    "y : array_like\n"
    "   The array of observations.\n"
    "err2 : array_like\n"
    "   The squared errors.\n\n"
    "Returns\n"
    "-------\n"
    "lnl : float\n"
    "   The log-likelihood.";

static char chisq_docstring[] = 
    "chisq(var, tcorr, t, y, err2)\n\n"
    "calculate the log-likelihood of a process with var and tcorr.\n\n"
    "Parameters\n"
    "----------\n"
    "var : float\n"
    "   The variance of the process.\n"
    "tcorr : float\n"
    "   The relaxation time of the process.\n"
    "t : array_like\n"
    "   The array of times.\n"
    "y : array_like\n"
    "   The array of observations.\n"
    "err2 : array_like\n"
    "   The squared errors.\n\n"
    "Returns\n"
    "-------\n"
    "chisq : float\n"
    "   The Chi-squared statistic.";

static PyObject *tridiagonal_trimult(PyObject *self, PyObject *args);
static PyObject *tridiagonal_trisolve(PyObject *self, PyObject *args);
static PyObject *tridiagonal_tridet(PyObject *self, PyObject *args);
static PyObject *tridiagonal_snsolve(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *tridiagonal_lnlike(PyObject *self, PyObject *args);
static PyObject *tridiagonal_chisq(PyObject *self, PyObject *args);


static PyMethodDef module_methods[] = {
    {"trimult", tridiagonal_trimult, METH_VARARGS, trimult_docstring},
    {"trisolve", tridiagonal_trisolve, METH_VARARGS, trisolve_docstring},
    {"tridet", tridiagonal_tridet, METH_VARARGS, tridet_docstring},
    {"snsolve", (PyCFunction)tridiagonal_snsolve, METH_VARARGS | METH_KEYWORDS, snsolve_docstring},
    {"lnlike", tridiagonal_lnlike, METH_VARARGS, lnlike_docstring},
    {"chisq", tridiagonal_chisq, METH_VARARGS, chisq_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC PyInit__tridiagonal(void)
{
    
    PyObject *module;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_tridiagonal",
        module_docstring,
        -1,
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };
    module = PyModule_Create(&moduledef);
    if (!module) return NULL;

    /* Load `numpy` functionality. */
    import_array();

    return module;
}

static PyObject *
tridiagonal_trimult(PyObject *self, PyObject *args)
{
    PyObject *a_obj, *b_obj, *c_obj, *x_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOOO", &a_obj, &b_obj, &c_obj, &x_obj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *a_array = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *b_array = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *c_array = PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (a_array == NULL || b_array == NULL || c_array == NULL || x_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(x_array);
        return NULL;
    }

    /* How many data points are there? */
    int n = (int)PyArray_DIM(x_array, 0);
    npy_intp dim[1] = {n};  
     


    /* Get pointers to the data as C-types. */
    double *a    = (double*)PyArray_DATA(a_array);
    double *b    = (double*)PyArray_DATA(b_array);
    double *c    = (double*)PyArray_DATA(c_array);
    double *x    = (double*)PyArray_DATA(x_array);
    
    /* Build output array */
    PyArrayObject *y_array = (PyArrayObject*)PyArray_SimpleNew(1, dim, NPY_DOUBLE);
    double *y = (double*)PyArray_DATA(y_array);

    /* Call the external C function to multiply. */
    trimult(a, b, c, x, y, n);

    /* Clean up. */
    Py_DECREF(x_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);

    /* Build the output tuple */
    return PyArray_Return(y_array);
}

static PyObject *
tridiagonal_trisolve(PyObject *self, PyObject *args)
{
    PyObject *a_obj, *b_obj, *c_obj, *x_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOOO", &a_obj, &b_obj, &c_obj, &x_obj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *a_array = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *b_array = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *c_array = PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (a_array == NULL || b_array == NULL || c_array == NULL || x_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(x_array);
        return NULL;
    }

    /* How many data points are there? */
    int n = (int)PyArray_DIM(x_array, 0);
    npy_intp dim[1] = {n};  
     


    /* Get pointers to the data as C-types. */
    double *a    = (double*)PyArray_DATA(a_array);
    double *b    = (double*)PyArray_DATA(b_array);
    double *c    = (double*)PyArray_DATA(c_array);
    double *x    = (double*)PyArray_DATA(x_array);
    
    /* Build output array */
    PyArrayObject *y_array = (PyArrayObject*)PyArray_SimpleNew(1, dim, NPY_DOUBLE);
    double *y = (double*)PyArray_DATA(y_array);

    /* Call the external C function to solve the system. */
    trisolve(a, b, c, y, x, n);

    /* Clean up. */
    Py_DECREF(x_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);

    /* Build the output tuple */
    return PyArray_Return(y_array);
}

static PyObject *
tridiagonal_tridet(PyObject *self, PyObject *args)
{
    PyObject *a_obj, *b_obj, *c_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOO", &a_obj, &b_obj, &c_obj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *a_array = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *b_array = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *c_array = PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (a_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return NULL;
    }

    /* How many data points are there? */
    int N = (int)PyArray_DIM(b_array, 0);

    /* Get pointers to the data as C-types. */
    double *a    = (double*)PyArray_DATA(a_array);
    double *b    = (double*)PyArray_DATA(b_array);
    double *c    = (double*)PyArray_DATA(c_array);

    /* Call the external C function to compute the log-determinant. */
    double value = tridet(a, b, c, N);

    /* Clean up. */
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);

    /* Build the output tuple */
    return Py_BuildValue("d", value);
}

static PyObject *
tridiagonal_snsolve(PyObject *self, PyObject *args, PyObject *kwargs)
{
    int ret_det = 0;
    double var, tcorr;
    PyObject *t_obj, *err2_obj, *y_obj;

    /* Parse the input tuple */
    static char *kwlist[] = {"var", "tcorr", "t_obj", "err2_obj", "y_obj", "ret_det", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ddOOO|i", kwlist, 
                                     &var, &tcorr, &t_obj, &err2_obj, &y_obj, &ret_det))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *t_array = PyArray_FROM_OTF(t_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *err2_array = PyArray_FROM_OTF(err2_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (y_array == NULL || t_array == NULL || err2_array == NULL) {
        Py_XDECREF(y_array);
        Py_XDECREF(t_array);
        Py_XDECREF(err2_array);
        return NULL;
    }

    /* How many data points are there? */
    int n = (int)PyArray_DIM(t_array, 0);
    npy_intp dim[1] = {n};  
    
    /* Get pointers to the data as C-types. */
    double *t    = (double*)PyArray_DATA(t_array);
    double *y    = (double*)PyArray_DATA(y_array);
    double *err2    = (double*)PyArray_DATA(err2_array);
    
    /* Build output array */
    PyArrayObject *x_array = (PyArrayObject*)PyArray_SimpleNew(1, dim, NPY_DOUBLE);
    double *x = (double*)PyArray_DATA(x_array);

    /* Call the external C function. */
    if (ret_det) {
        double ldetc = snsolve_retdet(t, err2, var, tcorr, x, y, n);

        /* Clean up. */
        Py_DECREF(t_array);
        Py_DECREF(err2_array);
        Py_DECREF(y_array);

        /* Build the output tuple */
        return Py_BuildValue("dO", ldetc, x_array);

    } 
    else {
        snsolve(t, err2, var, tcorr, x, y, n);

        /* Clean up. */
        Py_DECREF(t_array);
        Py_DECREF(err2_array);
        Py_DECREF(y_array);

        /* Return the array. */
        return PyArray_Return(x_array);
    }
}

static PyObject *
tridiagonal_lnlike(PyObject *self, PyObject *args)
{
    double var, tcorr;
    PyObject *t_obj, *err2_obj, *y_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "ddOOO", &var, &tcorr, &t_obj, &y_obj, &err2_obj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *t_array = PyArray_FROM_OTF(t_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *err2_array = PyArray_FROM_OTF(err2_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (y_array == NULL || t_array == NULL || err2_array == NULL) {
        Py_XDECREF(y_array);
        Py_XDECREF(t_array);
        Py_XDECREF(err2_array);
        return NULL;
    }

    /* How many data points are there? */
    int n = (int)PyArray_DIM(t_array, 0);
    
    /* Get pointers to the data as C-types. */
    double *t    = (double*)PyArray_DATA(t_array);
    double *y    = (double*)PyArray_DATA(y_array);
    double *err2    = (double*)PyArray_DATA(err2_array);
    
    /* Call the external C function. */

    double lnl = lnlike(var, tcorr, t, y, err2, n);

    /* Clean up. */
    Py_DECREF(t_array);
    Py_DECREF(err2_array);
    Py_DECREF(y_array);

    /* Build the output tuple */
    return Py_BuildValue("d", lnl);

}

static PyObject *
tridiagonal_chisq(PyObject *self, PyObject *args)
{
    double var, tcorr;
    PyObject *t_obj, *err2_obj, *y_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "ddOOO", &var, &tcorr, &t_obj, &y_obj, &err2_obj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *t_array = PyArray_FROM_OTF(t_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *err2_array = PyArray_FROM_OTF(err2_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (y_array == NULL || t_array == NULL || err2_array == NULL) {
        Py_XDECREF(y_array);
        Py_XDECREF(t_array);
        Py_XDECREF(err2_array);
        return NULL;
    }

    /* How many data points are there? */
    int n = (int)PyArray_DIM(t_array, 0);
    
    /* Get pointers to the data as C-types. */
    double *t    = (double*)PyArray_DATA(t_array);
    double *y    = (double*)PyArray_DATA(y_array);
    double *err2    = (double*)PyArray_DATA(err2_array);
    
    /* Call the external C function. */

    double val = chisq(var, tcorr, t, y, err2, n);

    /* Clean up. */
    Py_DECREF(t_array);
    Py_DECREF(err2_array);
    Py_DECREF(y_array);

    /* Build the output tuple */
    return Py_BuildValue("d", val);

}