"""
A variety of utility functions that are used across a wider framework. Only elements applicable to developing and
testing TheanoBatchDTW are present here and some changes from the original framework have been made to allow this code
to operate outside the rest of the framework. As a result of this approach, some of the code may not make a great deal
of sense!
"""

import numpy

import theano


def get_standard_dtype():
    """
    Returns theano.config.floatX. There is a reason for this in the framework but it is not applicable in
    TheanoBatchDTW (some code has been removed from this function).

    :return: theano.config.floatX
    """

    return theano.config.floatX


def shared(value, name, borrow=True):
    """
    Returns a theano shared variable with the given value. Unlike theano.shared, we assume borrowing can be enabled by
    default.

    :param value: The value to assign to the new shared variable (ownership of this value esentially passes to the
                  new shared variable -- no subsequent changes to this value should be made directly).
    :param name: The name to be given to the new shared variable.
    :param borrow: Whether the shared variable can use the value directly (True) or should take a deep copy for itself
                   (False)
    :return: The new shared variable.
    """

    if borrow:
        return theano.shared(value, name=name, borrow=borrow)
    return theano.shared(value, name=name)


def gaussian_random_matrix(rows, columns):
    """
    Returns a new numpy matrix of the specified shape and using the standard dtype. The matrix is filled with values
    obtained from numpy.random.randn (i.e. Gaussian distributed with zero mean and unit variance).

    :param rows: The number of rows in the new matrix.
    :param columns: The number of columns in the new matrix.
    :return: The new matrix filled with random values.
    """

    return numpy.random.randn(rows, columns).astype(get_standard_dtype())


def shared_gaussian_random_matrix(name, rows, columns):
    """
    Creates a new matrix filled with Gaussian distributed random variables, using gaussian_random_matrix, and
    immediately uses that matrix to initialize a new Theano shared variable.

    :param name: The name to be given to the new shared variable.
    :param rows: The number of rows in the new matrix.
    :param columns: The number of columns in the new matrix.
    :return: The new shared variable initialized with a matrix of Gaussian distributed random variables.
    """

    return shared(gaussian_random_matrix(rows, columns), name)
