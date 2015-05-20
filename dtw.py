#!/usr/bin/env python

"""
A testbed for developing a Theano batch DTW implementation. Computes DTW distances but does not currently output the
back-traced optimal paths. The intent of this implementation is not speed but gradients. The hope is to use Theano's
symbolic differentiation capability to compute the gradient of a cost function with respect to parameters where that
cost function might be "min DTW(x, y)" and where x and/or y are functions of the parameters.

The implementation works fully for individual sequence pairs but is 100 times slower than a reference Cython DTW
implementation. In principle it should be possible to submit batches of sequence pairs and compute the DTW distances for
the batch concurrently (e.g. on a GPU). In practice this implementation can compute the distances for a batch correctly
but fails to provide a correct gradient. NaNs and infs appear in the gradients where they should not and no method has
been found to eliminate these pesky intrusions.

NOTE: The DTW distances are correct even in batch mode. It's only the gradients which are wrong.

This file, and the dependencies in the framework folder, contains everything needed to exhibit the gradient problem.
With enable_grads=True applied in the _test_main call at the very end of this script you will experience exceptions
warning of unwanted NaNs.

The parameters in the _test_main call at the end of this script are explained here:

debug_level:
0: essentially off, though assertions still enabled
1: additional value checks enabled but general printing disabled.
2: All checks and printing enabled.

test_variations:
True: run the three DTW implementations (Cython reference, Theano non-batched, and Theano batched) many times using a
      variety of configurations and inputs. See the _test_main function to see which varieties are tried (or run the
      script!). Will take many minutes to run to completion.
False: only a couple of simpler configurations are run; intended for speeding up development and testing.

enable_grads:
True: Inputs are projected via a parameter matrix before being passed to DTW. Gradients of the loss function (min DTW)
      with respect to the parameter matrix are computed and tested. Currently not working correctly.
False: Inputs are passed directly to DTW without transformation. Gradients are not computed. For testing the plain
       DTW operation.

large_value: A value to use as if it were the same as positive infinity. numpy.inf works fine as long as gradients are
             not wanted. If gradients are wanted, may need to change this value to a large finite value to help prevent
             unwanted NaNs or infs appearing in the gradient (e.g. set to 1e300).

The Theano implementation is slow! While the Cython implementation can compute the DTW distance between 100 sequence
pairs in 0.079 seconds, the nonbatch Theano implementation takes 19.5 seconds and the fully batched Theano
implementation takes 0.5 seconds. Little attempt has been made to optimize either Theano implementation at this point as
the focus has been on getting the gradients to work.

Synthetic data is generated to test and compare the implementations. Simpler synthetic data is used as Theano test
values which, in combination with a custom Debug operation (see debug_op.py in the framework package), allows quick
detection of errors in the implementation.

The dynamic programming chart implemented here originates in the top-left, has a row for each position in x1, and a
column for each position in x2.
"""

import os
import time

os.environ['THEANO_FLAGS'] = 'device=cpu,openmp=False,floatX=float64'
# ,exception_verbosity=high
# ,optimizer=None

import numpy

import theano
import theano.compile
import theano.ifelse
import theano.printing
import theano.scan_module
import theano.scan_module.scan_op
import theano.tensor
import theano.tensor.extra_ops
import framework.debug_op
import framework.distance
import framework.cython_dtw
import framework.utility


def _debug(node, name, debug_level, check_not_all_nan=True, check_not_any_nan=True, check_not_all_inf=False,
           check_not_any_inf=False, raise_on_failed_nan_check=True, raise_on_failed_inf_check=True):
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
    return framework.debug_op.debug(node, name, debug_level, check_not_all_nan=check_not_all_nan,
                                    check_not_any_nan=check_not_any_nan, check_not_all_inf=check_not_all_inf,
                                    check_not_any_inf=check_not_any_inf,
                                    raise_on_failed_nan_check=raise_on_failed_nan_check,
                                    raise_on_failed_inf_check=raise_on_failed_inf_check)


def _create_dtw_inner_step(debug_level):
    """
    Creates a DTW inner step function given a debug level.

    :param debug_level: The debug level to use (see above for explanation).
    :return: A function that can be used inside theano.scan to compute the DTW inner steps.
    """

    def dtw_inner_step(distance, matching_cost, insertion_cost, deletion_cost):
        """
        The DTW inner step. To be called in a theano.scan operation that iterates over a single slice of distance(s)
        corresponding to a single temporal slice of x2. This is a symbolic operation: all parameters must be symbolic
        and the return value is symbolic.

        If x1 and x2 were 2D then all four parameters will be scalars; if x1 and x2 were 3D then the parameters will be
        vectors of size equal to the batch size (i.e. x1.shape[1] == x2.shape[1]).

        :param distance: The distance(s) for the current position in the DTW dynamic grid chart.
        :param matching_cost: The cost(s) computed in the previous DTW outer iteration and at the previous x1 position
                              (i.e. the cost(s) found moving one step up and one step left in the DTW dynamic
                              programming chart).
        :param insertion_cost: The cost(s) computed in the previous DTW outer iteration at the same x1 offset (i.e. the
                               cost(s) found moving one step left in the DTW dynamic programming chart).
        :param deletion_cost: The previous cost(s) in the current x2 slice (i.e. the cost(s) found moving one step up in
                              the DTW dynamic programming chart).
        :return: The result of computing "distance + min(matching_cost, insertion_cost, deletion_cost)".
        """

        assert distance.ndim == matching_cost.ndim == insertion_cost.ndim == deletion_cost.ndim
        distance = _debug(distance, 'dtw_inner_step.distance', debug_level)
        matching_cost = _debug(matching_cost, 'dtw_inner_step.matching_cost', debug_level)
        insertion_cost = _debug(insertion_cost, 'dtw_inner_step.insertion_cost', debug_level)
        deletion_cost = _debug(deletion_cost, 'dtw_inner_step.deletion_cost', debug_level)
        min_cost = _debug(theano.tensor.min(theano.tensor.stack(matching_cost, insertion_cost, deletion_cost), axis=0),
                          'dtw_inner_step.min_cost', debug_level)
        assert min_cost.ndim == distance.ndim
        cost = _debug(distance + min_cost, 'dtw_inner_step.cost', debug_level)
        assert cost.ndim == distance.ndim
        return cost

    return dtw_inner_step


def _create_dtw_outer_step(distance_function, large_value, debug_level):
    """
    Creates a DTW outer step function given requested configuration settings.

    :param distance_function: A symbolic function for computing the distance between sequence elements such as those
                              found in framework.distance (e.g. cosine, Euclidean, etc.).
    :param large_value: A real scalar value (already been cast to the appropriate dtype) that denotes infinite distance.
                        In principle we would like this to be numpy.inf but is parameterised here in case this padding
                        value is contributing to the NaN/inf-gradient problem.
    :param debug_level: The debug level to use (see above for explanation).
    :return: A function that can be used inside theano.scan to compute the DTW outer steps.
    """

    assert distance_function is not None
    assert large_value is not None
    dtype = framework.utility.get_standard_dtype()

    def dtw_outer_step(x2_index, x2_temporal_slice, previous_costs, previous_best_costs, x1, x1_lengths, x2_lengths):
        """
        The DTW outer step. To be called in a theano.scan operation that iterates over all temporal slices of x2. This
        is a symbolic operation: all parameters must be symbolic and the return value is symbolic.

        :param x2_index: The index of the x2 slice being processed in this iteration. Used to identify x2 sequences that
                         are padded out to the necessary tensor size.
        :param x2_temporal_slice: The slice of x2 being processed in this iteration.
        :param previous_costs: The costs (which may contain padding values) from the previous iteration, used for
                               matching and insertion costs inthis iteration.
        :param previous_best_costs: The best costs found so far. Not affected by inf padding.
        :param x1: The whole x1 matrix against which the current x2 slice will be compared when computing distances.
        :param x1_lengths: The lengths of the x1 sequences in the current batch (to identify where padding is used).
        :param x2_lengths: The lengths of the x2 sequences in the current batch (to identify where padding is used).
        :return: The costs computed for the current x2 slice and the potentially updated best costs for each item in the
                 batch.
        """

        ndim = x1.ndim
        assert 2 <= ndim == (x2_temporal_slice.ndim + 1) <= 3

        # Debug wrap the variable inputs
        x2_index = _debug(x2_index, 'dtw_outer_step.x2_index', debug_level)
        x2_temporal_slice = _debug(x2_temporal_slice, 'dtw_outer_step.x2_temporal_slice', debug_level)
        previous_costs = _debug(previous_costs, 'dtw_outer_step.previous_costs', debug_level)
        previous_best_costs = _debug(previous_best_costs, 'dtw_outer_step.previous_best_costs', debug_level)

        # Compute distances between the single-frame slice of x2 and all frames in x1
        x2_temporal_slice = theano.tensor.stack(x2_temporal_slice)
        distances = _debug(distance_function(x1, x2_temporal_slice), 'dtw_outer_step.distances.1', debug_level,
                           check_not_any_nan=False)

        # Mask out the distances for shorter x1 sequences
        mask = _debug(
            (theano.tensor.zeros_like(distances, dtype='int32').T + theano.tensor.arange(x1.shape[0])).T >= x1_lengths,
            'dtw_outer_step.mask1', debug_level)
        distances = _debug(
            theano.tensor.set_subtensor(distances[theano.tensor.nonzero(mask)], large_value),
            'dtw_outer_step.distances.2', debug_level, check_not_any_nan=False)

        # Mask out the distances for shorter x2 sequences (and init costs)
        if ndim == 2:
            distances = _debug(
                theano.tensor.switch(x2_lengths <= x2_index, large_value, distances),
                'dtw_outer_step.distances.3a', debug_level)
            large_values = large_value
        elif ndim == 3:
            mask = _debug(x2_index >= (theano.tensor.zeros_like(distances, dtype='int32') + x2_lengths),
                          'dtw_outer_step.mask2', debug_level)
            distances = _debug(
                theano.tensor.set_subtensor(distances[theano.tensor.nonzero(mask)], large_value),
                'dtw_outer_step.distances.3b', debug_level)
            large_values = theano.tensor.zeros_like(x1[0, :, 0], dtype=dtype) + large_value
        else:
            raise Exception('Unsupported number of dimensions: ' + str(ndim))

        # Execute the inner step for each of the distances just computed. See dtw_inner_step for an explanation of the
        # other sequences and initial output values.
        results, _ = theano.scan(_create_dtw_inner_step(debug_level),
                                 sequences=[distances, previous_costs[:-1], previous_costs[1:]],
                                 outputs_info=[large_values])

        # Prepend the costs we've just computed with large values to initialize the matching and insertion costs in the
        # next outer step iteration.
        costs = _debug(theano.tensor.concatenate([[large_values], results]), 'dtw_outer_step.costs', debug_level)

        # Identify the best cost(s). We assume non-batch inputs are not padded.
        if ndim == 2:
            best_costs = results[-1]
        elif ndim == 3:
            best_costs = theano.tensor.switch(x2_lengths <= x2_index, previous_best_costs,
                                              results[x1_lengths - 1, theano.tensor.arange(distances.shape[1])])
        else:
            raise Exception('Unsupported number of dimensions: ' + str(ndim))

        best_costs = _debug(best_costs, 'dtw_outer_step.best_costs', debug_level, check_not_any_inf=True)
        return costs, best_costs

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
    c = _debug(theano.ifelse.ifelse(comparison, b, a), 'theano_symbolic_dtw.%s' % name_a, debug_level)
    d = _debug(theano.ifelse.ifelse(comparison, a, b), 'theano_symbolic_dtw.%s' % name_b, debug_level)
    return c, d


def theano_symbolic_dtw(x1, x2, x1_lengths, x2_lengths, distance_function=framework.distance.cosine, normalize=True,
                        large_value=1e300, debug_level=None):
    """
    A symbolic implementation of DTW that supports batches of sequence pairs.

    Returns a scalar if ndim == 2 and a vector of size x1.shape[1] if ndim == 3

    This is slow! About 90 times slower than the Cython implementation using the parameters below.

    :param x1: A tensor containing the first side of the sequence pairs to be aligned.
    :param x2: A tensor containing the second side of the sequence pairs to be aligned.
    :param x1_lengths: An integer vector identifying the lengths of the sequences in x1
    :param x2_lengths: An integer vector identifying the lengths of the sequences in x2
    :param distance_function: The symbolic distance function to use (e.g. a reference to a function in
                              framework.distance).
    :param normalize: Whether the DTW distances should be sequence length normalized.
    :param large_value: A value to stand in for positive infinity when padding the distances tensor. Can be numpy.inf if
                        you don't care about gradients.
    :param debug_level: The debug level to use (see above for explanation).
    :return: The DTW distances for every sequence pair in the batch.
    """

    assert 0 <= x1_lengths.ndim == x2_lengths.ndim <= 1
    assert isinstance(normalize, bool)

    ndim = x1.ndim
    assert 2 <= ndim == x2.ndim <= 3

    # Ensure x2 is the shorter input to minimize the number of scan iterations
    x1_length, x2_length = x1.shape[0], x2.shape[0]
    x1_shorter_than_x2 = theano.tensor.le(x1_length, x2_length)

    x1, x2 = _swap(x1_shorter_than_x2, x1, x2, 'x1', 'x2', debug_level)
    x1_lengths, x2_lengths = _swap(x1_shorter_than_x2, x1_lengths, x2_lengths, 'x1_lengths', 'x2_lengths', debug_level)
    x1_length, x2_length = _swap(x1_shorter_than_x2, x1_length, x2_length, 'x1_length', 'x2_length', debug_level)

    dtype = framework.utility.get_standard_dtype()
    large_value = numpy.dtype(dtype).type(large_value)
    zero = numpy.dtype(dtype).type(0.)

    if ndim == 2:
        # Dimensions: temporal position, embedding position
        initial_best_costs = zero
        initial_costs = theano.tensor.zeros_like(x1[:, 0], dtype=dtype) + large_value
    elif ndim == 3:
        # Dimensions: temporal position, batch position, embedding position
        initial_best_costs = theano.tensor.zeros_like(x1[0, :, 0], dtype=dtype)
        initial_costs = theano.tensor.zeros_like(x1[:, :, 0], dtype=dtype) + large_value
    else:
        raise Exception('Unsupported number of dimensions: ' + str(ndim))

    indexes = _debug(theano.tensor.arange(x2_length), 'theano_symbolic_dtw.indexes', debug_level)
    initial_costs = theano.tensor.concatenate([[initial_best_costs], initial_costs])

    # Iterate over the temporal slices of x2. See dtw_outer_step for an explanation of the other inputs to this scan
    # operation
    results, _ = theano.scan(
        _create_dtw_outer_step(distance_function, large_value, debug_level),
        sequences=[indexes, x2], outputs_info=[initial_costs, initial_best_costs],
        non_sequences=[x1, x1_lengths, x2_lengths])
    result = _debug(results[-1][-1], 'theano_symbolic_dtw.result', debug_level, check_not_any_inf=True)

    # Length normalize the distances if requested to do so
    if normalize:
        result = _debug(result / theano.tensor.cast(x1_lengths + x2_lengths, dtype=dtype),
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
                  framework.utility.get_standard_dtype.
    :param test_value_getter: A method for generating test values. Defaults to zero mean unit variance Gaussian values.
    :return: The newly created Theano variable.
    """

    if dtype is None:
        dtype = framework.utility.get_standard_dtype()

    if len(test_value_shape) == 0:
        x = theano.tensor.scalar(name, dtype=dtype)
    elif len(test_value_shape) == 1:
        x = theano.tensor.vector(name, dtype=dtype)
    elif len(test_value_shape) == 2:
        x = theano.tensor.matrix(name, dtype=dtype)
    elif len(test_value_shape) == 3:
        x = theano.tensor.tensor3(name, dtype=dtype)
    else:
        raise Exception('Unsupported number of dimensions: ' + str(len(test_value_shape)))

    if debug_level > 0:
        x.tag.test_value = test_value_getter(test_value_shape)

        if len(test_value_shape) == 0:
            x.tag.test_value = numpy.dtype(dtype).type(x.tag.test_value)
        else:
            x.tag.test_value = x.tag.test_value.astype(dtype)

    return x, _debug(x, '%s.%s' % (debug_name_prefix, name), debug_level)


def _test_theano_compiled_dtw(input_size, hidden_size, ndim, distance_function, normalize, enable_grads,
                              large_value, debug_level):
    """
    Performs a test of a Theano DTW implementation.

    :param input_size: The size of the inputs.
    :param hidden_size: The size of the hidden values (used only if enable_grads=True).
    :param ndim: The number of dimensions to use (2: non-batched, 3: batched).
    :param distance_function: The symbolic distance function to use (e.g. a reference to a function in
                              framework.distance).
    :param normalize: Whether the DTW distances should be sequence length normalized.
    :param enable_grads: Whether gradients should be computed of a min mean DTW cost function with respect to some
                         synthetic parameters.
    :param large_value: A value to stand in for positive infinity when padding the distances tensor. Can be numpy.inf if
                        you don't care about gradients.
    :param debug_level: The debug level to use (see above for explanation).
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

    # Construct the symbolic expression for DTW
    symbolic_dtw = theano_symbolic_dtw(x1, x2, x1_lengths, x2_lengths, distance_function=distance_function,
                                       normalize=normalize, large_value=large_value, debug_level=debug_level)
    outputs = [symbolic_dtw]

    if enable_grads:
        # Create some synthetic parameters
        w = framework.utility.shared_gaussian_random_matrix('w', input_size, hidden_size)

        # Transform the inputs using the synthetic parameters
        z1 = _debug(theano.dot(x1, w), 'theano_compiled_dtw.z1', debug_level)
        z2 = _debug(theano.dot(x2, w), 'theano_compiled_dtw.z2', debug_level)

        # Create a new DTW expression using the transformed inputs
        symbolic_dtw = theano_symbolic_dtw(z1, z2, x1_lengths, x2_lengths, distance_function=distance_function,
                                           normalize=normalize, large_value=large_value,
                                           debug_level=debug_level)

        # Create a min mean DTW cost expression
        cost = _debug(theano.tensor.mean(symbolic_dtw) if ndim == 3 else symbolic_dtw, 'theano_compiled_dtw.cost',
                      debug_level, check_not_any_inf=True)
        outputs.append(cost)

        # Perform symbolic differentiation of the cost expression with respect to the synthetic parameters
        outputs.append(_debug(theano.grad(cost, w), 'theano_compiled_dtw.w_grad', debug_level, check_not_any_inf=True))

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
    dtype = framework.utility.get_standard_dtype()

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
            return list(outputs[0])

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

    return _cython_dtw(batch, normalize, framework.cython_dtw.multivariate_dtw_cost_cosine)


def cython_dtw_euclidean(batch, normalize=True):
    """
    A variant of _cython_dtw that uses the Euclidean distance function between sequence elements.

    :param batch: The batch of inputs whose DTW distances are to be computed.
    :param normalize: Whether DTW distances should be length normalized.
    :return: The resulting DTW distances.
    """

    return _cython_dtw(batch, normalize, framework.cython_dtw.multivariate_dtw_cost_euclidean)


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


def _test_theano_make_dtw_function(input_size, hidden_size, distance_mode, ndim, normalize, enable_grads, large_value,
                                   debug_level):
    """
    Used during testing. A version of _test_common_make_dtw_function that is specific to the Theano implementations.

    :param input_size: The size of the inputs.
    :param hidden_size: The size of the hidden values (used only if enable_grads=True).
    :param distance_mode: A string used to lookup the requested DTW function (typically either 'cosine' or 'euclidean').
    :param ndim: The number of dimensions to use (2: non-batched, 3: batched).
    :param normalize: Whether DTW distances should be length normalized.
    :param enable_grads: Whether gradients should be computed of a min mean DTW cost function with respect to some
                         synthetic parameters.
    :param large_value: A value to stand in for positive infinity when padding the distances tensor. Can be numpy.inf if
                        you don't care about gradients.
    :param debug_level: The debug level to use (see above for explanation).
    :return: The requested DTW function implementation.
    """

    def compile_distance_function(distance_function):
        compiled_distance_function = _test_theano_compiled_dtw(input_size, hidden_size, ndim, distance_function,
                                                               normalize, enable_grads, large_value, debug_level)
        return lambda batch: theano_dtw(batch, compiled_distance_function, ndim == 3)

    return _test_common_make_dtw_function(
        distance_mode, normalize, dict(cosine=lambda: compile_distance_function(framework.distance.cosine),
                                       euclidean=lambda: compile_distance_function(framework.distance.euclidean)))


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

    return [(framework.utility.gaussian_random_matrix(numpy.random.randint(min_x1_length, max_x1_length), input_size),
             framework.utility.gaussian_random_matrix(numpy.random.randint(min_x2_length, max_x2_length), input_size))
            for _ in xrange(batch_size)]


def _test_case(distance_mode, normalize, iterations, debug_level, min_x1_length, max_x1_length, min_x2_length,
               max_x2_length, batch_size, input_size, hidden_size, enable_grads, large_value):
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
    :param large_value: A value to stand in for positive infinity when padding the distances tensor. Can be numpy.inf if
                        you don't care about gradients.
    :return: None
    """
    print 'distance_mode: %s, normalize: %s, iterations: %s, debug_level: %s, min_x1_length: %s, max_x1_length: %s, ' \
          'min_x2_length: %s, max_x2_length: %s, batch_size: %s, input_size: %s, hidden_size: %s, enable_grads: %s, ' \
          'large_value: %s' % (
              distance_mode, normalize, iterations, debug_level, min_x1_length, max_x1_length, min_x2_length,
              max_x2_length, batch_size, input_size, hidden_size, enable_grads, large_value)
    numpy.random.seed(1)

    dtw_functions = (
        _test_cython_make_dtw_function(distance_mode, normalize),
        _test_theano_make_dtw_function(input_size, hidden_size, distance_mode, 2, normalize, enable_grads, large_value,
                                       debug_level),
        _test_theano_make_dtw_function(input_size, hidden_size, distance_mode, 3, normalize, enable_grads, large_value,
                                       debug_level))

    numpy.random.seed(1)

    all_inputs = [
        _test_rand_inputs(min_x1_length, max_x1_length, min_x2_length, max_x2_length, batch_size, input_size) for _ in
        xrange(iterations)]
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
            assert numpy.allclose(result_a, result_b), (result_a, result_b)


def _test_main(debug_level, test_variations, enable_grads, large_value):
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
    :param large_value: A value to use as if it were the same as positive infinity. numpy.inf works fine as long as
                        gradients are not wanted. If gradients are wanted, may need to change this value to a large
                        finite value to help prevent unwanted NaNs or infs appearing in the gradient (e.g. set to
                        1e300).
    :return: None
    """
    os.environ['__THEANO_IMPORT_debug_level'] = str(debug_level)

    if debug_level > 0:
        theano.config.compute_test_value = 'raise'

    if test_variations:
        for distance_mode in ['cosine', 'euclidean']:
            for normalize in [True, False]:
                for iterations, batch_size in [(100, 1), (10, 10), (1, 100)]:
                    _test_case(distance_mode=distance_mode, normalize=normalize, iterations=iterations,
                               debug_level=debug_level, min_x1_length=80, max_x1_length=100, min_x2_length=110,
                               max_x2_length=130, batch_size=batch_size, input_size=39, hidden_size=39,
                               enable_grads=enable_grads, large_value=large_value)
    else:
        _test_case(distance_mode='cosine', normalize=True, iterations=2, debug_level=debug_level, min_x1_length=5,
                   max_x1_length=10, min_x2_length=10, max_x2_length=15, batch_size=2, input_size=3, hidden_size=4,
                   enable_grads=enable_grads, large_value=large_value)
        _test_case(distance_mode='euclidean', normalize=True, iterations=2, debug_level=debug_level, min_x1_length=5,
                   max_x1_length=10, min_x2_length=10, max_x2_length=15, batch_size=2, input_size=3, hidden_size=4,
                   enable_grads=enable_grads, large_value=large_value)


if __name__ == '__main__':
    print 'THEANO_FLAGS', os.environ['THEANO_FLAGS']
    _test_main(debug_level=0, test_variations=True, enable_grads=False, large_value=numpy.inf)
