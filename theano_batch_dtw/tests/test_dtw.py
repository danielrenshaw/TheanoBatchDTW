import numpy
import speech_dtw._dtw

import theano
import theano.tensor as tt
import theano_batch_dtw.dtw


def test_dtw():
    W = theano.shared(numpy.eye(4, dtype=theano.config.floatX), name='W')
    theano.config.compute_test_value = 'raise'
    x1 = tt.matrix('x1')
    x2 = tt.matrix('x2')
    x1.tag.test_value = numpy.array([[0.1] * 4, [-0.1] * 4], dtype=theano.config.floatX)
    x2.tag.test_value = numpy.array([[0.1] * 4, [0.1] * 1 + [-0.1] * 3, [-0.1] * 4], dtype=theano.config.floatX)
    e1 = theano.dot(x1, W)
    e2 = theano.dot(x2, W)
    y = theano_batch_dtw.dtw.theano_symbolic_dtw(e1, e2, tt.constant(2, dtype='int64'), tt.constant(3, dtype='int64'),
                                                 normalize=False)
    theano.printing.debugprint(y)
    g = theano.grad(y, W)
    theano.printing.debugprint(g)
    print 'y', y.dtype, y.tag.test_value.shape, '\n', y.tag.test_value
    print 'g', g.dtype, g.tag.test_value.shape, '\n', g.tag.test_value
    path, cost = speech_dtw._dtw.multivariate_dtw(e1.tag.test_value, e2.tag.test_value)
    print cost, list(reversed(path))


test_dtw()
