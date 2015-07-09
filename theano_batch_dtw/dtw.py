#!/usr/bin/env python

"""
A testbed for developing a Theano batch DTW implementation. Computes DTW distances but does not currently output the
back-traced optimal paths. The intent of this implementation is not speed but gradients. The hope is to use Theano's
symbolic differentiation capability to compute the gradient of a cost function with respect to parameters where that
cost function might be "min DTW(x1, x2)" and where x1 and/or x2 are functions of the parameters.

The implementation works fully for individual sequence pairs but is 100 times slower than a reference Cython DTW
implementation. Better speed is achieved by computing many sequence pair DTWs in a (mini-)batch mode. Typical timings,
in seconds, for computing the DTW between 1000 sequence pairs is:

 * Cython: 0.5813
 * Theano 2D: 58.86
 * Theano 3D, batch size=1: 187.6
 * Theano 3D, batch size=10: 24.23
 * Theano 3D, batch size=100: 4.778
 * Theano 3D, batch size=1000: 2.628

Note that the 3D (batching) version is slower than the 2D version when the batch size is too small. Little attempt has
been made to optimize either Theano implementation as the focus has been on getting the gradients to work.

To produce your own timings, just run this script. By default gradients are disabled. The code slows down substantially
when gradients are enabled but that is not a fair comparison with the Cython version since that is incapable of
computing gradients.

The parameters in the _test_main call at the end of this script are explained here:

debug_level:
0: essentially off, though assertions still enabled if enable_value_checks=True
1: additional value checks enabled but general printing disabled.
2: All checks and printing enabled.

test_variations:
True: run the three DTW implementations (Cython reference, Theano non-batched, and Theano batched) many times using a
      variety of configurations and inputs. See the _test_main function to see which varieties are tried (or run the
      script!). Will take many minutes to run to completion.
False: only a couple of simpler configurations are run; intended for speeding up development and testing.

enable_grads:
True: Inputs are projected via a parameter matrix before being passed to DTW. Gradients of the loss function (min DTW)
      with respect to the parameter matrix are computed and tested.
False: Inputs are passed directly to DTW without transformation. Gradients are not computed. For testing the plain
       DTW operation.

enable_value_checks:
True: Checks that the values produced by the three implementations are close (via numpy.allclose). Should normally be
      enabled but available to be disabled to aid development and debugging of temporarily invalid variations.
False: Values checks are disabled.

eps:
Previous versions of this code computed the DTW values correctly but failed to compute the gradients. The problem
manifested itself as NaN values in the gradients. The problem was tracked down to division by zero which only occurred
in the gradient computation graph seemingly outside the bounds of the divide by zero checks in the code proper. The
solution is to use a minimum value inside the sqrt operation of the magnitude function and in the computation of the
divisor in the cosine distance function. By default this value equals the machine epsilon for the Python float data
type.


Synthetic data is generated to test and compare the implementations. Simpler synthetic data is used as Theano test
values which, in combination with a custom Debug operation (see debug_op.py in the framework package), allows quick
detection of errors in the implementation.

The dynamic programming chart implemented here originates in the top-left, has a row for each position in x1, and a
column for each position in x2.
"""

import os
import time

if __name__ == '__main__':
    # os.environ['THEANO_FLAGS'] = 'device=cpu,openmp=True,floatX=float64'
    os.environ['THEANO_FLAGS'] = 'device=cpu,openmp=True,floatX=float32,warn_float64=raise'
    # os.environ['THEANO_FLAGS'] = 'device=cpu,openmp=False,floatX=float32,warn_float64=raise,compute_test_value=raise,' \
    #                              'on_opt_error=raise,on_shape_error=raise,numpy.seterr_all=warn,optimizer=None,' \
    #                              'exception_verbosity=high'

import numpy

import theano
import theano.compile
import theano.ifelse
import theano.printing
import theano.scan_module
import theano.scan_module.scan_op
import theano.tensor as tt
import theano.tensor.extra_ops
import debug_op
import cython_dtw
import utility

DTYPE_INT64 = 'int64'


def _debug(node, name, debug_level, check_not_all_nan=True, check_not_any_nan=True, check_not_all_inf=True,
           check_not_any_inf=True, raise_on_failed_nan_check=True, raise_on_failed_inf_check=True):
    """
    Allows the default parameters for the framework's debug operation to be altered without editing any other users.
    The "check" parameters only affect the expression, not the expression's gradient.

    :param node: The node to be debugged.
    :param name: The name of the operation being debugging. The node's name will be changed to this.
    :param debug_level: The debug level with zero being the lowest and None means debugging will be fully off.
    :param check_not_all_nan: Whether the node's output should be checked to ensure it contains some non-NaN values.
    :param check_not_any_nan: Whether the node's output should be checked to ensure it doesn't contain any NaN values.
    :param check_not_all_inf: Whether the node's output should be checked to ensure it contains some non-inf values.
    :param check_not_any_inf: Whether the node's output should be checked to ensure it doesn't contain any inf values.
    :param raise_on_failed_nan_check: Whether an exception should be raised if a NaN check fails.
    :param raise_on_failed_inf_check: Whether an exception should be raised if an inf check fails.
    :return: The Debug wrapped node.
    """
    return debug_op.debug(node, name, debug_level, check_not_all_nan=check_not_all_nan,
                          check_not_any_nan=check_not_any_nan, check_not_all_inf=check_not_all_inf,
                          check_not_any_inf=check_not_any_inf, raise_on_failed_nan_check=raise_on_failed_nan_check,
                          raise_on_failed_inf_check=raise_on_failed_inf_check)


def _create_dtw_inner_step(debug_level):
    """
    Creates a DTW inner step function given a debug level.

    :param debug_level: The debug level to use (see above for explanation).
    :return: A function that can be used inside theano.scan to compute the DTW inner steps.
    """

    def dtw_inner_step(x2_index, d_slice_slice, insert_cost, x1_length, x2_length, x1_index, previous_cost_row):
        assert x2_index.ndim == 0
        assert 0 <= d_slice_slice.ndim <= 1
        assert insert_cost.ndim == d_slice_slice.ndim
        assert x1_length.ndim == d_slice_slice.ndim
        assert x2_length.ndim == d_slice_slice.ndim
        assert x1_index.ndim == 0
        assert previous_cost_row.ndim == d_slice_slice.ndim + 1

        x2_index = _debug(x2_index, 'dtw_inner_step.x2_index', debug_level)
        d_slice_slice = _debug(d_slice_slice, 'dtw_inner_step.d_slice_slice', debug_level)
        insert_cost = _debug(insert_cost, 'dtw_inner_step.insert_cost', debug_level)

        delete_cost = _debug(previous_cost_row[x2_index], 'dtw_inner_step.delete_cost', debug_level)
        match_cost = _debug(previous_cost_row[x2_index - 1], 'dtw_inner_step.match_cost', debug_level)
        assert delete_cost.ndim == d_slice_slice.ndim
        assert match_cost.ndim == d_slice_slice.ndim

        min_cost = _debug(tt.min(tt.stack(insert_cost, delete_cost, match_cost), axis=0), 'dtw_inner_step.min_cost',
                          debug_level)
        assert min_cost.ndim == d_slice_slice.ndim

        in_first_row = _debug(tt.eq(x1_index, 0), 'dtw_inner_step.in_first_row', debug_level)
        in_first_column = _debug(tt.eq(x2_index, 0), 'dtw_inner_step.in_first_column', debug_level)
        assert in_first_row.ndim == 0
        assert in_first_column.ndim == 0

        cost = _debug(
            d_slice_slice + tt.switch(in_first_row, insert_cost, tt.switch(in_first_column, delete_cost, min_cost)),
            'dtw_inner_step.cost', debug_level)
        assert cost.ndim == d_slice_slice.ndim

        length_filtered_cost = _debug(
            tt.switch(tt.bitwise_and(tt.lt(x1_index, x1_length), tt.lt(x2_index, x2_length)), cost, 0.),
            'dtw_inner_step.length_filtered_cost', debug_level)
        assert length_filtered_cost.ndim == d_slice_slice.ndim

        return length_filtered_cost

    return dtw_inner_step


def _create_dtw_outer_step(distance_function, debug_level):
    """
    Creates a DTW outer step function given requested configuration settings.

    :param distance_function: A symbolic function for computing the distance between sequence elements such as those
                              found in distance (e.g. cosine, Euclidean, etc.).
    :param debug_level: The debug level to use (see above for explanation).
    :return: A function that can be used inside theano.scan to compute the DTW outer steps.
    """

    assert distance_function is not None

    def dtw_outer_step(x1_index, d_slice, previous_cost_row, x1_length, x2_length):
        assert x1_index.ndim == 0
        assert 1 <= d_slice.ndim <= 2
        assert previous_cost_row.ndim == d_slice.ndim
        assert x1_length.ndim == d_slice.ndim - 1
        assert x2_length.ndim == d_slice.ndim - 1

        x1_index = _debug(x1_index, 'dtw_outer_step.x1_index', debug_level)
        d_slice = _debug(d_slice, 'dtw_outer_step.d_slice', debug_level)
        previous_cost_row = _debug(previous_cost_row, 'dtw_outer_step.previous_cost_row', debug_level)

        x2_indexes = tt.arange(d_slice.shape[0], dtype=DTYPE_INT64)
        results, _ = theano.scan(_create_dtw_inner_step(debug_level), sequences=[x2_indexes, d_slice],
                                 outputs_info=[tt.zeros_like(d_slice[0], dtype=theano.config.floatX)],
                                 non_sequences=[x1_length, x2_length, x1_index, previous_cost_row])
        return results

    return dtw_outer_step


def _swap(comparison, a, b, name_a, name_b, debug_level):
    """
    Symbolically swaps a and b if the symbolic comparison is true. Also wraps the results in Debug operations.

    :param comparison: Swaps the two items if this is symbolically True.
    :param a: The first symbolic item.
    :param b: The second symbolic item.
    :param name_a: The name of the resulting first item (i.e. after the potential swapping).
    :param name_b: The name of the resulting second item (i.e. after the potential swapping).
    :param debug_level: The debug level to use (see above for explanation).
    :return: The two Debug wrapped items which may have been swapped.
    """
    c = _debug(theano.ifelse.ifelse(comparison, b, a), '_swap.%s' % name_a, debug_level)
    d = _debug(theano.ifelse.ifelse(comparison, a, b), '_swap.%s' % name_b, debug_level)
    return c, d


def transpose_and_pad(x1, x2):
    return x1.dimshuffle(range(x1.ndim - 1, 0, -1) + ['x', 0]), x2.dimshuffle(range(x2.ndim - 1, -1, -1) + ['x'])


def magnitude(x, eps):
    # We must use a minimum value inside the sqrt to avoid a NaN in the gradient
    return tt.sqrt(tt.maximum(tt.sqr(x).sum(x.ndim - 1), eps))


def euclidean(x1, x2, eps):
    assert x1.ndim == x2.ndim
    x1, x2 = x1.dimshuffle([0, 'x'] + range(1, x1.ndim)), x2.dimshuffle(['x'] + range(x2.ndim))
    result = magnitude(x1 - x2, eps)
    assert result.ndim == x1.ndim - 1
    return result


def cosine(x1, x2, eps):
    assert x1.ndim == x2.ndim
    magnitudes = transpose_and_pad(magnitude(x1, eps), magnitude(x2, eps))
    # We must use a minimum value to avoid a NaN in the gradient
    divisor = tt.maximum(magnitudes[0] * magnitudes[1], eps)
    x1, x2 = transpose_and_pad(x1, x2)
    result = tt.switch(tt.neq(divisor, 0.), 1. - (x1 * x2).sum(axis=0) / divisor, 0.).T
    assert result.ndim == x1.ndim - 1
    return result


def theano_symbolic_dtw(x1, x2, x1_lengths, x2_lengths, distance_function=cosine, normalize=True, debug_level=None,
                        eps=None):
    """
    A symbolic implementation of DTW that supports batches of sequence pairs.

    Returns a scalar if ndim == 2 and a vector of size x1.shape[1] if ndim == 3

    This is slow! About 90 times slower than the Cython implementation using the parameters below.

    :param x1: A tensor containing the first side of the sequence pairs to be aligned.
    :param x2: A tensor containing the second side of the sequence pairs to be aligned.
    :param x1_lengths: An integer vector identifying the lengths of the sequences in x1
    :param x2_lengths: An integer vector identifying the lengths of the sequences in x2
    :param distance_function: The symbolic distance function to use (e.g. a reference to a function in
                              distance).
    :param normalize: Whether the DTW distances should be sequence length normalized.
    :param debug_level: The debug level to use (see above for explanation).
    :param eps: The minimum value to use inside the distance function. Set to the machine epsilon if None.
    :return: The DTW distances for every sequence pair in the batch.
    """

    if eps is None:
        eps = numpy.dtype(theano.config.floatX).type(numpy.finfo(float).eps)

    assert 0 <= x1_lengths.ndim == x2_lengths.ndim <= 1
    assert isinstance(normalize, bool)

    ndim = x1.ndim
    assert 2 <= ndim == x2.ndim <= 3

    # Ensure x2 is the shorter input to minimize the number of scan iterations
    x1_shorter_than_x2 = tt.le(x1.shape[0], x2.shape[0])
    x1, x2 = _swap(x1_shorter_than_x2, x1, x2, 'x1', 'x2', debug_level)
    x1_lengths, x2_lengths = _swap(x1_shorter_than_x2, x1_lengths, x2_lengths, 'x1_lengths', 'x2_lengths', debug_level)

    # Compute distances between x1 sequences and paired x2 sequences
    d = distance_function(x1, x2, eps)

    # Iterate over the temporal slices of x2. See dtw_outer_step for an explanation of the other inputs to this scan
    # operation
    x1_indexes = tt.arange(x1.shape[0], dtype=DTYPE_INT64)
    results, _ = theano.scan(_create_dtw_outer_step(distance_function, debug_level), sequences=[x1_indexes, d],
                             outputs_info=[
                                 tt.zeros_like(x2[:, :, 0] if x2.ndim == 3 else x2[:, 0], dtype=theano.config.floatX)],
                             non_sequences=[x1_lengths, x2_lengths])
    result = results[x1_lengths - 1, x2_lengths - 1, tt.arange(x1.shape[1])] if x2.ndim == 3 else results[
        x1_lengths - 1, x2_lengths - 1]
    result = _debug(result, 'theano_symbolic_dtw.result', debug_level)
    assert result.ndim == x1_lengths.ndim

    # Length normalize the distances if requested to do so
    if normalize:
        result = _debug(result / tt.cast(x1_lengths + x2_lengths, dtype=utility.get_standard_dtype()),
                        'theano_symbolic_dtw.norm_result', debug_level)

    return result


def _common_dtw(batch, dtw):
    """
    Used during testing. Performs various checks on the inputs to a DTW operation, executes the DTW operation, and
    performs various checks on the output of the DTW operation.

    :param batch: The batch of sequence pairs to be passed to the DTW function.
    :param dtw: The DTW function to be executed.
    :return: The results after checks have verified their validity.
    """

    assert batch is not None

    if not isinstance(batch, (tuple, list)):
        batch = [batch]
    else:
        assert len(batch) > 0

    for sequence_pair in batch:
        assert isinstance(sequence_pair, (tuple, list))
        assert len(sequence_pair) == 2
        x1, x2 = sequence_pair
        assert isinstance(x1, numpy.ndarray)
        assert isinstance(x2, numpy.ndarray)
        assert x1.ndim == x2.ndim == 2

    results = dtw(batch)
    assert isinstance(results, (tuple, list))

    for result in results:
        assert isinstance(result, float), type(result)

    return tuple(results)


def _var(name, test_value_shape, debug_name_prefix, debug_level, dtype=None,
         test_value_getter=lambda shape: numpy.random.randn(*shape)):
    """
    Creates a new symbolic variable with the given name and generates a synthetic test value of the requested shape. The
    resulting Theano variable is wrapped in a Debug operation.

    :param name: The name of the variable to be created.
    :param test_value_shape: The shape of the test value.
    :param debug_name_prefix: Used in naming the Debug operation.
    :param debug_level: The debug level to use (see above for explanation).
    :param dtype: The type of the variable being created. Defaults to whatever is returned by
                  utility.get_standard_dtype.
    :param test_value_getter: A method for generating test values. Defaults to zero mean unit variance Gaussian values.
    :return: The newly created Theano variable.
    """

    if dtype is None:
        dtype = utility.get_standard_dtype()

    if len(test_value_shape) == 0:
        x = tt.scalar(name, dtype=dtype)
    elif len(test_value_shape) == 1:
        x = tt.vector(name, dtype=dtype)
    elif len(test_value_shape) == 2:
        x = tt.matrix(name, dtype=dtype)
    elif len(test_value_shape) == 3:
        x = tt.tensor3(name, dtype=dtype)
    else:
        raise Exception('Unsupported number of dimensions: ' + str(len(test_value_shape)))

    if debug_level > 0:
        x.tag.test_value = test_value_getter(test_value_shape)

        if len(test_value_shape) == 0:
            x.tag.test_value = numpy.dtype(dtype).type(x.tag.test_value)
        else:
            x.tag.test_value = x.tag.test_value.astype(dtype)

    return x, _debug(x, '%s.%s' % (debug_name_prefix, name), debug_level)


def _test_theano_compiled_dtw(input_size, hidden_size, ndim, distance_function, normalize, enable_grads, debug_level,
                              eps):
    """
    Performs a test of a Theano DTW implementation.

    :param input_size: The size of the inputs.
    :param hidden_size: The size of the hidden values (used only if enable_grads=True).
    :param ndim: The number of dimensions to use (2: non-batched, 3: batched).
    :param distance_function: The symbolic distance function to use (e.g. a reference to a function in
                              distance).
    :param normalize: Whether the DTW distances should be sequence length normalized.
    :param enable_grads: Whether gradients should be computed of a min mean DTW cost function with respect to some
                         synthetic parameters.
    :param debug_level: The debug level to use (see above for explanation).
    :param eps: The minimum value to use inside the distance function. Set to the machine epsilon if None.
    :return: A compiled Theano function that can be used to compute DTW distances between sequence pairs.
    """

    assert 2 <= ndim <= 3

    # Create the input variables test values and lengths suitable for testing the implementation.
    if ndim == 2:
        x1_in, x1 = _var('x1', (4, input_size), 'theano_compiled_dtw', debug_level)
        x2_in, x2 = _var('x2', (5, input_size), 'theano_compiled_dtw', debug_level)
        x1_lengths_in, x1_lengths = _var('x1_lengths', (), 'theano_compiled_dtw', debug_level, dtype='int32',
                                         test_value_getter=lambda shape: 4)
        x2_lengths_in, x2_lengths = _var('x2_lengths', (), 'theano_compiled_dtw', debug_level, dtype='int32',
                                         test_value_getter=lambda shape: 5)
    elif ndim == 3:
        x1_in, x1 = _var('x1', (5, 4, input_size), 'theano_compiled_dtw', debug_level)
        x2_in, x2 = _var('x2', (6, 4, input_size), 'theano_compiled_dtw', debug_level)
        if debug_level > 0:
            x1.tag.test_value[-1, 0] = 0
            x2.tag.test_value[-1, 1] = 0
            x1.tag.test_value[-1, 2] = 0
            x2.tag.test_value[-1, 2] = 0
        x1_lengths_in, x1_lengths = _var('x1_lengths', (2,), 'theano_compiled_dtw', debug_level, dtype='int32',
                                         test_value_getter=lambda shape: numpy.array([4, 5, 4, 5]))
        x2_lengths_in, x2_lengths = _var('x2_lengths', (2,), 'theano_compiled_dtw', debug_level, dtype='int32',
                                         test_value_getter=lambda shape: numpy.array([6, 5, 5, 6]))
    else:
        raise Exception('Unsupported number of dimensions: ' + str(ndim))

    if enable_grads:
        # Create some synthetic parameters
        w = utility.shared_gaussian_random_matrix('w', input_size, hidden_size)

        # Transform the inputs using the synthetic parameters
        x1 = _debug(theano.dot(x1, w), 'theano_compiled_dtw.z1', debug_level)
        x2 = _debug(theano.dot(x2, w), 'theano_compiled_dtw.z2', debug_level)
    else:
        w = None

    # Construct the symbolic expression for DTW
    symbolic_dtw = theano_symbolic_dtw(x1, x2, x1_lengths, x2_lengths, distance_function=distance_function,
                                       normalize=normalize, debug_level=debug_level, eps=eps)
    outputs = [symbolic_dtw]

    if enable_grads:
        # Create a min mean DTW cost expression
        cost = _debug(tt.mean(symbolic_dtw) if ndim == 3 else symbolic_dtw, 'theano_compiled_dtw.cost', debug_level)
        outputs.append(cost)

        # Perform symbolic differentiation of the cost expression with respect to the synthetic parameters
        outputs.append(_debug(theano.grad(cost, w), 'theano_compiled_dtw.w_grad', debug_level))

    return theano.function([x1_in, x2_in, x1_lengths_in, x2_lengths_in], outputs, name='compiled_dtw_' + str(ndim),
                           on_unused_input='ignore')


def _check_outputs(outputs):
    """
    Used to check the additional outputs of a Theano DTW test, in particular the gradients if present. Currently only
    checks that nothing is NaN or inf.

    :param outputs: The outputs to be checked.
    :return: None
    """

    # We skip ove rhte first output which is the DTW distances that are checked elsewhere more generally.
    for output_index, output in enumerate(outputs[1:]):
        output = numpy.array(output)
        assert not numpy.isnan(output).any(), (output_index, output)
        assert not numpy.isinf(output).any(), (output_index, output)


def theano_dtw(batch, compiled_dtw, pack_batch=True):
    """
    Calls the provided compiled Theano DTW function using the appropriate mechanims (batching inputs when supported else
    submitted the sequence pairs one-at-a-time).

    :param batch: The sequence pairs whose DTW distances are to be calculated.
    :param compiled_dtw: The compiled DTW function to use.
    :param pack_batch: Whether the batch should be packed or not (i.e. the input dimensionality supported by
                       compiled_dtw). If pack_batch=True then compiled_dtw must accept 3D (batched) inputs else
                       compiled_dtw must accept 2D (non-batched) inputs.
    :return: The DTW distances between all the sequence pairs in batch.
    """

    assert compiled_dtw is not None
    assert isinstance(pack_batch, bool)
    dtype = utility.get_standard_dtype()

    def dtw(validated_batch):
        if pack_batch:
            # A validated batch input to this method is still in "list of pairs" format. To use the batched DTW function
            # we must pack those pairs into a single 3D tensor. The tensor size is determined by the maximum sequence
            # lengths on each side of the pairs. All shorter sequences are zero padded to fill the tensor. An integer
            # array is constructed to inform the DTW function how long the sequences are on each side.
            max_lengths = [max(lengths) for lengths in zip(*((len(x1), len(x2)) for x1, x2 in validated_batch))]
            batch_size = len(validated_batch)
            input_size = validated_batch[0][0].shape[1]
            packed_x1 = numpy.zeros((max_lengths[0], batch_size, input_size), dtype=dtype)
            packed_x2 = numpy.zeros((max_lengths[1], batch_size, input_size), dtype=dtype)
            x1_lengths = numpy.empty(batch_size, dtype='int32')
            x2_lengths = numpy.empty(batch_size, dtype='int32')

            for index, (x1, x2) in enumerate(validated_batch):
                packed_x1[:len(x1), index, :] = x1
                packed_x2[:len(x2), index, :] = x2
                x1_lengths[index] = len(x1)
                x2_lengths[index] = len(x2)

            outputs = compiled_dtw(packed_x1, packed_x2, x1_lengths, x2_lengths)
            _check_outputs(outputs)
            return [float(output) for output in outputs[0]]

        # If we're not packing the batch then we need to submit them one at a time.
        results = []
        for x1, x2 in validated_batch:
            outputs = compiled_dtw(x1, x2, len(x1), len(x2))
            _check_outputs(outputs)
            results.append(float(outputs[0]))
        return results

    return _common_dtw(batch, dtw)


def _cython_dtw(batch, normalize, multivariate_dtw_cost):
    """
    Executes the specified Cython DTW implementation with the specified batch of inputs. Each sequence pair is submitted
    one at a time.

    :param batch: The batch of inputs whose DTW distances are to be computed.
    :param normalize: Whether DTW distances should be length normalized.
    :param multivariate_dtw_cost: The specific DTW implementation to use.
    :return: The resulting DTW distances.
    """

    assert isinstance(normalize, bool)

    def dtw(validated_batch):
        return [multivariate_dtw_cost(x1, x2, normalize) for x1, x2 in validated_batch]

    return _common_dtw(batch, dtw)


def cython_dtw_cosine(batch, normalize=True):
    """
    A variant of _cython_dtw that uses the cosine distance function between sequence elements.

    :param batch: The batch of inputs whose DTW distances are to be computed.
    :param normalize: Whether DTW distances should be length normalized.
    :return: The resulting DTW distances.
    """

    return _cython_dtw(batch, normalize, cython_dtw.multivariate_dtw_cost_cosine)


def cython_dtw_euclidean(batch, normalize=True):
    """
    A variant of _cython_dtw that uses the Euclidean distance function between sequence elements.

    :param batch: The batch of inputs whose DTW distances are to be computed.
    :param normalize: Whether DTW distances should be length normalized.
    :return: The resulting DTW distances.
    """

    return _cython_dtw(batch, normalize, cython_dtw.multivariate_dtw_cost_euclidean)


def _test_common_make_dtw_function(distance_mode, normalize, dtw_functions):
    """
    Used during testing to construct (if needed) the DTW function for the requested distance mode.

    :param distance_mode: A string used to lookup the requested DTW function (typically either 'cosine' or 'euclidean').
    :param normalize: Whether DTW distances should be length normalized.
    :param dtw_functions: The dictionary of DTW function constructors to use to get the requested DTW function
                          implementation.
    :return: The requested DTW function implementation.
    """

    assert isinstance(distance_mode, str)
    assert isinstance(normalize, bool)
    assert isinstance(dtw_functions, dict)
    assert len(dtw_functions) > 0
    assert distance_mode in dtw_functions
    return dtw_functions[distance_mode]()


def _test_cython_make_dtw_function(distance_mode, normalize):
    """
    Used during testing. A version of _test_common_make_dtw_function that is specific to the Cython implementations.

    :param distance_mode: A string used to lookup the requested DTW function (typically either 'cosine' or 'euclidean').
    :param normalize: Whether DTW distances should be length normalized.
    :return: The requested DTW function implementation.
    """

    return _test_common_make_dtw_function(distance_mode, normalize,
                                          dict(cosine=lambda: lambda batch: cython_dtw_cosine(batch, normalize),
                                               euclidean=lambda: lambda batch: cython_dtw_euclidean(batch, normalize)))


def _test_theano_make_dtw_function(input_size, hidden_size, distance_mode, ndim, normalize, enable_grads, debug_level,
                                   eps):
    """
    Used during testing. A version of _test_common_make_dtw_function that is specific to the Theano implementations.

    :param input_size: The size of the inputs.
    :param hidden_size: The size of the hidden values (used only if enable_grads=True).
    :param distance_mode: A string used to lookup the requested DTW function (typically either 'cosine' or 'euclidean').
    :param ndim: The number of dimensions to use (2: non-batched, 3: batched).
    :param normalize: Whether DTW distances should be length normalized.
    :param enable_grads: Whether gradients should be computed of a min mean DTW cost function with respect to some
                         synthetic parameters.
    :param debug_level: The debug level to use (see above for explanation).
    :param eps: The minimum value to use inside the distance function. Set to the machine epsilon if None.
    :return: The requested DTW function implementation.
    """

    def compile_distance_function(distance_function):
        compiled_distance_function = _test_theano_compiled_dtw(input_size, hidden_size, ndim, distance_function,
                                                               normalize, enable_grads, debug_level, eps)
        return lambda batch: theano_dtw(batch, compiled_distance_function, ndim == 3)

    return _test_common_make_dtw_function(distance_mode, normalize,
                                          dict(cosine=lambda: compile_distance_function(cosine),
                                               euclidean=lambda: compile_distance_function(euclidean)))


def _test_rand_inputs(min_x1_length, max_x1_length, min_x2_length, max_x2_length, batch_size, input_size):
    """
    Used during testing. Constructs a batch of synthetic sequence pairs according to the provided specification.

    :param min_x1_length: The minimum length of sequence in x1.
    :param max_x1_length: The maximum length of sequence in x1.
    :param min_x2_length: The minimum length of sequence in x2.
    :param max_x2_length: The maximum length of sequence in x2.
    :param batch_size: The number of sequence pairs to be generated for this batch.
    :param input_size: The size of the sequence elements.
    :return: The list of pairs containing the synthetic data for the requested batch.
    """

    return [(utility.gaussian_random_matrix(numpy.random.randint(min_x1_length, max_x1_length), input_size),
             utility.gaussian_random_matrix(numpy.random.randint(min_x2_length, max_x2_length), input_size)) for _ in
            xrange(batch_size)]


def _test_case(distance_mode, normalize, iterations, debug_level, min_x1_length, max_x1_length, min_x2_length,
               max_x2_length, batch_size, input_size, hidden_size, enable_grads, enable_value_checks, eps):
    """
    Executes a general test case. All three DTW implementations will be tested in the given configuration.

    The synthetic data is generated using a fixed seed so the data is consistent across tests and across executions.

    :param distance_mode: A string used to lookup the requested DTW function (typically either 'cosine' or 'euclidean').
    :param normalize: Whether DTW distances should be length normalized.
    :param iterations: The number of batches.
    :param debug_level: The debug level to use (see above for explanation).
    :param min_x1_length: The minimum length of sequence in x1.
    :param max_x1_length: The maximum length of sequence in x1.
    :param min_x2_length: The minimum length of sequence in x2.
    :param max_x2_length: The maximum length of sequence in x2.
    :param batch_size: The size of each batch.
    :param input_size: The size of the sequence elements.
    :param hidden_size: The size of the hidden values (used only if enable_grads=True).
    :param enable_grads: Whether gradients should be computed of a min mean DTW cost function with respect to some
                         synthetic parameters.
    :param enable_value_checks: Whether numpy.allclose should be used to verify the values computed by the different DTW
                                implementations.
    :param eps: The minimum value to use inside the distance function. Set to the machine epsilon if None.
    :return: None
    """
    print 'distance_mode: %s, normalize: %s, iterations: %s, debug_level: %s, min_x1_length: %s, max_x1_length: %s, ' \
          'min_x2_length: %s, max_x2_length: %s, batch_size: %s, input_size: %s, hidden_size: %s, enable_grads: %s, ' \
          'enable_value_checks: %s, eps: %s' % (
              distance_mode, normalize, iterations, debug_level, min_x1_length, max_x1_length, min_x2_length,
              max_x2_length, batch_size, input_size, hidden_size, enable_grads, enable_value_checks, eps)
    numpy.random.seed(1)

    dtw_functions = (_test_cython_make_dtw_function(distance_mode, normalize),
                     _test_theano_make_dtw_function(input_size, hidden_size, distance_mode, 2, normalize, enable_grads,
                                                    debug_level, eps),
                     _test_theano_make_dtw_function(input_size, hidden_size, distance_mode, 3, normalize, enable_grads,
                                                    debug_level, eps))

    numpy.random.seed(1)

    all_inputs = [_test_rand_inputs(min_x1_length, max_x1_length, min_x2_length, max_x2_length, batch_size, input_size)
                  for _ in xrange(iterations)]
    all_results = [[] for _ in xrange(iterations)]

    for function_index, dtw_function in enumerate(dtw_functions):
        if debug_level > 1:
            print 'Function', function_index
        start = time.clock()

        for inputs_index, inputs in enumerate(all_inputs):
            if debug_level > 1:
                print 'Inputs', inputs_index, inputs
            all_results[inputs_index].append(dtw_function(inputs))

        print 'time %f' % (time.clock() - start,)

    for results_index, results in enumerate(all_results):
        results = [numpy.asarray(result) for result in results]
        if debug_level > 1:
            print 'Results', results_index, results
        for result_a, result_b in zip(results[:-1], results[1:]):
            if enable_value_checks:
                assert numpy.allclose(result_a, result_b), (result_a, result_b)


def _test_main(debug_level, test_variations, enable_grads, enable_value_checks, eps):
    """
    Executes a sequence of test cases.

    :param debug_level: 0: essentially off, though assertions still enabled. 1: additional value checks enabled but
                        general printing disabled. 2: All checks and printing enabled.
    :param test_variations: True: run the three DTW implementations (Cython reference, Theano non-batched, and Theano
                            batched) many times using a variety of configurations and inputs. See the _test_main
                            function to see which varieties are tried (or run the script!). Will take many minutes to
                            run to completion. False: only a couple of simpler configurations are run; intended for
                            speeding up development and testing.
    :param enable_grads: True: Inputs are projected via a parameter matrix before being passed to DTW. Gradients of the
                               loss function (min DTW) with respect to the parameter matrix are computed and tested.
                               Currently not working correctly. False: Inputs are passed directly to DTW without
                               transformation. Gradients are not computed. For testing the plain DTW operation.
    :param enable_value_checks: Whether numpy.allclose should be used to verify the values computed by the different DTW
                                implementations.
    :param eps: The minimum value to use inside the distance function. Set to the machine epsilon if None.
    :return: None
    """
    os.environ['__THEANO_IMPORT_debug_level'] = str(debug_level)

    if debug_level > 0:
        theano.config.compute_test_value = 'raise'

    if test_variations:
        for distance_mode in ['cosine', 'euclidean']:
            for normalize in [True, False]:
                for iterations, batch_size in [(1000, 1), (100, 10), (10, 100), (1, 1000)]:
                    _test_case(distance_mode=distance_mode, normalize=normalize, iterations=iterations,
                               debug_level=debug_level, min_x1_length=80, max_x1_length=100, min_x2_length=110,
                               max_x2_length=130, batch_size=batch_size, input_size=39, hidden_size=39,
                               enable_grads=enable_grads, enable_value_checks=enable_value_checks, eps=eps)
    else:
        _test_case(distance_mode='cosine', normalize=True, iterations=2, debug_level=debug_level, min_x1_length=5,
                   max_x1_length=10, min_x2_length=10, max_x2_length=15, batch_size=2, input_size=3, hidden_size=4,
                   enable_grads=enable_grads, enable_value_checks=enable_value_checks, eps=eps)
        _test_case(distance_mode='euclidean', normalize=True, iterations=2, debug_level=debug_level, min_x1_length=5,
                   max_x1_length=10, min_x2_length=10, max_x2_length=15, batch_size=2, input_size=3, hidden_size=4,
                   enable_grads=enable_grads, enable_value_checks=enable_value_checks, eps=eps)


if __name__ == '__main__':
    print 'THEANO_FLAGS', os.environ['THEANO_FLAGS']
    _test_main(debug_level=1, test_variations=False, enable_grads=True, enable_value_checks=False, eps=None)
