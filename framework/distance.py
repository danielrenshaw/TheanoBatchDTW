"""
A variety of Theano symbolic distance functions. Each should operate (transparently) with batched inputs or individual
inputs.

Part of a wider framework. Only elements applicable to developing and testing TheanoBatchDTW are present here and some
changes from the original framework may have been made to allow this code to operate outside the rest of the framework.
"""

import numpy

import theano
import theano.tensor
import utility


def magnitude(x):
    """
    Computes the Euclidean norm of x along the last dimension of x.

    Not itself a distance function, but a common component of distance functions.

    :param x: The data whose mahnitude(s) are to be computed.
    :return: A tensor of one dimension smaller than x containing the computed magnitudes.
    """
    return theano.tensor.sqrt(theano.tensor.sqr(x).sum(x.ndim - 1))


def euclidean(x, y):
    """
    The Euclidean distance between x and y along the final dimensions of x and y. x and y need to have matching shapes
    (given a valid broadcasting).

    :param x: A point, or set of points, in a vector space determined by the final dimension of x.
    :param y: A point, or set of points, in a vector space determined by the final dimension of y.
    :return: The Euclidean distance between the points in x and corresponding points in y, has dimensionality one less
             than x and y.
    """
    assert x.ndim == y.ndim
    result = magnitude(x - y)
    assert result.ndim == x.ndim - 1
    return result


def cosine(x, y):
    """
    The cosine distance between x and y along the final dimensions of x and y. x and y need to have matching shapes
    (given a valid broadcasting).

    :param x: A point, or set of points, in a vector space determined by the final dimension of x.
    :param y: A point, or set of points, in a vector space determined by the final dimension of y.
    :return: The cosine distance between the points in x and corresponding points in y, has dimensionality one less
             than x and y.
    """
    assert x.ndim == y.ndim
    one = numpy.dtype(utility.get_standard_dtype()).type(1.)
    result = one - (x * y).sum(x.ndim - 1) / (magnitude(x) * magnitude(y))
    assert result.ndim == x.ndim - 1
    return result
