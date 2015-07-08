"""
Wraps a Cython implementation of DTW. We use this as a reference implementation in testing and debugging the Theano
implementation. To use an alternate reference implementation you might consider altering these wrapper functions only.

The Cython implementation used here, speech_dtw, can be found at https://github.com/kamperh/speech_dtw.
"""

import numpy
import speech_dtw._dtw


def multivariate_dtw_cost_cosine(s, t, dur_normalize=False):
    return speech_dtw._dtw.multivariate_dtw_cost_cosine(s.astype(numpy.float64), t.astype(numpy.float64),
                                                        dur_normalize=dur_normalize)


def multivariate_dtw_cost_euclidean(s, t, dur_normalize=False):
    return speech_dtw._dtw.multivariate_dtw_cost_euclidean(s.astype(numpy.float64), t.astype(numpy.float64),
                                                           dur_normalize=dur_normalize)
