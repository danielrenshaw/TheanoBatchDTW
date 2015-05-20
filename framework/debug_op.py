"""
A Theano op that operates like an identity operation (input values are passed through unaltered and emitted as this
operation's output values). To that extent, it works very much like the theano.printing.Print operation. Unlike the
built in Print op, this op also allows for R_op and gradients to printed/interrogated. The nature of the printing is
also different from theano.printing.Print in that test_values are emitted when present integrating similar operations
that are useful during debugging of expression construction with those at function execution time.
"""

import re

import numpy

import theano
import theano.gof


class Debug(theano.gof.Op):
    view_map = {0: [0]}

    __props__ = ('name', 'debug_level', 'check_not_all_nan', 'check_not_any_nan', 'check_not_all_inf',
                 'check_not_any_inf', 'raise_on_failed_nan_check', 'raise_on_failed_inf_check')

    def _check(self, check_not_all, check_not_any, masker, value, prefix):
        """
        Checks for and returns information about the presence of certain values. Typically used to check for NaN or inf
        values.

        :param check_not_all: Whether we should check if the mask covers the entire tensor.
        :param check_not_any: Whether we should check if the mask covers any of the tensor.
        :param masker: A function we can call to mask out values that are not of interest to this check operation.
        :param value: The tensor whose values are to be checked.
        :param prefix: A string prefix identifying the type of check being performed.
        :return: A tuple consisting of a string with information about the extent of the special value mask over the
                 value, and a boolean indicating whether either of the checks (all/any) "failed" (i.e. returned True).
        """

        if self.debug_level > 0 or check_not_all or check_not_any:
            mask = masker(value)

            if mask.all():
                info = 'all'
                failed_check = check_not_all
            elif mask.any():
                info = 'some'
                failed_check = check_not_any
            else:
                info = 'none'
                failed_check = False

            return '<%s: %s>' % (prefix, info), failed_check

    def _test_exception(self, exception, return_exception):
        """
        Many of the check functions can be called with a parameter that changes their exception handling behaviour. This
        function pulls that code out into a common place.

        If exception is None then None is returned, otherwise exception is returned if return_exception is True,
        otherwise exception is raised.

        :param exception: The exception to be returned or raised (but may be None).
        :param return_exception: A boolean indicating whether a non-None exception should be returned (True) or raised
                                 (False)
        :return: exception if return_exception else None
        """

        assert self is not None  # Hack to stop PyCharm complaining about potential static method

        if exception is None:
            return None

        if return_exception:
            return exception

        raise exception

    def _action_check(self, check_failed, name, check_type, node, force_print, return_exception):
        """
        Actions an earlier "check" test. If the check failed (or printing has been forced on), will print detailed
        information about provided node. May raise a new exception (or return it if requested) if the check failed.

        :param check_failed: A boolean indicating whether the earlier check failed or not.
        :param name: The name of the operation/position where the check failed.
        :param check_type: A string indicating the type of check that failed (e.g. 'NaN' or 'inf').
        :param node: The node whose details are to be printed if the check failed.
        :param force_print: Even if the check succeeded we can force printing of the additional information if that
                            information is useful for other debugging purposes.
        :param return_exception: Whether an exception, if there is one, should be returned (True) or raised (False)
        :return: The exception if one exists and return_exception is True, otherwise None.
        """

        exception = None

        if check_failed or force_print:
            if node is not None:
                print '*** %s pp:' % name
                print theano.printing.pp(node)
                print '*** %s debugprint (all):' % name
                print theano.printing.debugprint(node, ids='id', print_type=True, include_nan_info=True,
                                                 include_inf_info=True, print_test_value=True)
                print '*** %s debugprint (limited):' % name
                print theano.printing.debugprint(node, ids='id', print_type=True, include_nan_info=True,
                                                 include_inf_info=True,
                                                 recursion_rules='SOME_NANS,ALL_NANS,NO_TEST_VALUE',
                                                 print_test_value=True)

            exception = Exception('Failed %s %s check' % (name, check_type))

        return self._test_exception(exception, return_exception)

    def _print_value(self, value, name, node=None, enable_all_checks=False, disable_all_checks=False, force_print=False,
                     return_exception=False):
        """
        Prints details about a given value if the debug level is high enough, if NaN/inf checks fail, or if printing has
        been forced on.

        :param value: The value whose details are to be potentially printed.
        :param name: The name of the location where this value has been obtained from.
        :param node: The node most pertinent to the value (i.e. output value).
        :param enable_all_checks: Whether all NaN and inf checks should be forced on.
        :param disable_all_checks: Whether all NaN and inf checks should be forced off.
        :param force_print: Whether printing should be forced on.
        :param return_exception: Whether a check failure should result in an exception being returned (True) or raised
                                 (False).
        :return: The exception if one exists and return_exception is True, otherwise None.
        """

        exception = None

        if force_print or self.debug_level > 0:
            check_not_all_nan = enable_all_checks or self.check_not_all_nan
            check_not_any_nan = enable_all_checks or self.check_not_any_nan
            check_not_all_inf = enable_all_checks or self.check_not_all_inf
            check_not_any_inf = enable_all_checks or self.check_not_any_inf
            nan_info, nan_check_failed = self._check(check_not_all_nan, check_not_any_nan, numpy.isnan, value, 'nan')
            inf_info, inf_check_failed = self._check(check_not_all_inf, check_not_any_inf, numpy.isinf, value, 'inf')

            if disable_all_checks:
                nan_check_failed = False
                inf_check_failed = False

            if force_print or nan_check_failed or inf_check_failed or self.debug_level > 1:
                name = '%s.%s' % (self.name, name)

                if isinstance(value, numpy.ndarray):
                    type_info = '<type: %s %s>' % (value.dtype, value.shape)
                else:
                    type_info = type(value)

                print '%s %s %s %s' % (name, type_info, nan_info, inf_info),
                print re.sub('\\s+', ' ', repr(value)) if self.debug_level > 1 else ''

                exception = self._action_check(self.raise_on_failed_nan_check and nan_check_failed, name, 'nan', node,
                                               force_print, return_exception) if exception is None else exception
                exception = self._action_check(self.raise_on_failed_inf_check and inf_check_failed, name, 'inf', node,
                                               force_print, return_exception) if exception is None else exception

        return self._test_exception(exception, return_exception)

    def _print_test_value(self, node, name, enable_all_checks=False, disable_all_checks=False, force_print=False,
                          return_exception=False):
        """
        Prints details about a given node's test value if it exists and if the debug level is high enough, if NaN/inf
        checks fail, or if printing has been forced on.

        :param node: The node whose test value is to be printed.
        :param name: The name of the location where this node has come from.
        :param enable_all_checks: Whether all NaN and inf checks should be foced on.
        :param disable_all_checks: Whether all NaN and inf checks should be forced off.
        :param force_print: Whether printing should be forced on.
        :param return_exception: Whether a check failure should result in an exception being returned (True) or raised
                                 (False).
        :return: The exception if one exists and return_exception is True, otherwise None.
        """

        exception = None

        if (force_print or self.debug_level > 0) and hasattr(node, 'tag') and hasattr(node.tag, 'test_value'):
            exception = self._print_value(node.tag.test_value, name + '.test_value', node=node,
                                          enable_all_checks=enable_all_checks, disable_all_checks=disable_all_checks,
                                          force_print=force_print, return_exception=return_exception)

        return self._test_exception(exception, return_exception)

    def _print_test_values(self, nodes, parent_name, name, other_nodes=None, enable_all_checks=False,
                           disable_all_checks=False):
        """
        Prints details about a given set of node's test values if they exist and if the debug level is high enough, if
        NaN/inf checks fail, or if printing has been forced on.

        :param nodes: The nodes whose test values are to be printed.
        :param parent_name: The name prefix of the location where these nodes have come from.
        :param name: The name suffix of the location where these nodes have come from.
        :param enable_all_checks: Whether all NaN and inf checks should be foced on.
        :param disable_all_checks: Whether all NaN and inf checks should be forced off.
        """

        exception = None

        for node_index, node in enumerate(nodes):
            exception = self._print_test_value(node, '%s.%s.%s' % (parent_name, name, node_index),
                                               enable_all_checks=enable_all_checks,
                                               disable_all_checks=disable_all_checks,
                                               return_exception=True) if exception is None else exception

        if exception is not None and other_nodes is not None:
            if not isinstance(other_nodes, (tuple, list)):
                other_nodes = [other_nodes]

            for other_node_index, other_node in enumerate(other_nodes):
                self._print_test_value(other_node, '%s.%s.%s' % (parent_name, name, other_node_index), force_print=True,
                                       return_exception=True)

        self._test_exception(exception, False)

    def __init__(self, name, debug_level, check_not_all_nan=True, check_not_any_nan=False, check_not_all_inf=True,
                 check_not_any_inf=False, raise_on_failed_nan_check=False, raise_on_failed_inf_check=False):
        self.name = name
        self.debug_level = debug_level
        self.check_not_all_nan = check_not_all_nan
        self.check_not_any_nan = check_not_any_nan
        self.check_not_all_inf = check_not_all_inf
        self.check_not_any_inf = check_not_any_inf
        self.raise_on_failed_nan_check = raise_on_failed_nan_check
        self.raise_on_failed_inf_check = raise_on_failed_inf_check
        super(Debug, self).__init__()

    def make_node(self, input_node):
        assert not isinstance(input_node, (tuple, list))
        # No need to print test value here because, if test values are enabled, "perform" will be called with the test
        # value as input. If this comment is wrong, could use the following line here, but may produce duplicate output.
        # self._print_test_value(input_node, 'make_node.input_node')
        return theano.gof.Apply(op=self, inputs=[input_node], outputs=[input_node.type.make_variable()])

    def perform(self, node, input_values, output_storage):
        input_value = input_values[0]
        output_storage[0][0] = input_value
        self._print_value(input_value, 'perform.input_value', node=node.inputs[0])

    def grad(self, input_nodes, output_gradients):
        # We cannot be sure that input or output gradients will avoid nans and infs, even if the expressions being
        # debugging by this instance cannot themselves generate nans or infs, hence the use of disable_all_checks.
        self._print_test_values(input_nodes, 'grad', 'input_node', other_nodes=output_gradients,
                                disable_all_checks=True)
        self._print_test_values(output_gradients, 'grad', 'output_gradient', other_nodes=input_nodes,
                                disable_all_checks=True)
        return output_gradients

    def R_op(self, input_nodes, eval_points):
        self._print_test_values(input_nodes, 'R_op', 'input_node', other_nodes=eval_points)
        self._print_test_values(eval_points, 'R_op', 'eval_point', other_nodes=input_nodes)
        return eval_points

    def __setstate__(self, dct):
        self.__dict__.update(dct)

    def c_code_cache_version(self):
        return 1,


def debug(node, name, debug_level, check_not_all_nan=True, check_not_any_nan=False, check_not_all_inf=True,
          check_not_any_inf=False, raise_on_failed_nan_check=True, raise_on_failed_inf_check=True):
    """
    A function that simply wraps the provided node in a Debug operation using the specified configuration settings.

    :param node: The node to be wrapped in a Debug operation.
    :param name: The name of the debug operation. The name of the node will also be changed to this value.
    :param debug_level: The debug level to use for this Debug operation.
    :param check_not_all_nan: Whether we should check if NaNs appear everywhere in the node's output.
    :param check_not_any_nan: Whether we should check if NaNs appear anywhere in the node's output.
    :param check_not_all_inf: Whether we should check if infs appear everywhere in the node's output.
    :param check_not_any_inf: Whether we should check if infs appear anywhere in the node's output.
    :param raise_on_failed_nan_check: Whether a failed NaN check should cause an exception to be raised or just
                                      returned.
    :param raise_on_failed_inf_check: Whether a failed inf check should cause an exception to be raised or just
                                      returned.
    :return: The node wrapped in a Debug operation (and with its name changed).
    """

    node.name = name
    result = Debug(name, debug_level, check_not_all_nan, check_not_any_nan, check_not_all_inf, check_not_any_inf,
                   raise_on_failed_nan_check, raise_on_failed_inf_check)(node)
    result.name = name
    return result
