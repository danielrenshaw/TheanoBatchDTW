# TheanoBatchDTW
A batch (multiple concurrent sequence pairs) implementation of Dynamic Time Warping (DTW) in Theano.

Contains a testbed for developing a Theano batch DTW implementation. Computes DTW distances but does not currently
output the back-traced optimal paths. The intent of this implementation is not speed but gradients. The hope is to use
Theano's symbolic differentiation capability to compute the gradient of a cost function with respect to parameters where
that cost function might be "min DTW(x, y)" and where x and/or y are functions of the parameters.

The implementation works fully for individual sequence pairs but is 100 times slower than a reference Cython DTW
implementation. In principle it should be possible to submit batches of sequence pairs and compute the DTW distances for
the batch concurrently (e.g. on a GPU). In practice this implementation can compute the distances for a batch correctly
but fails to provide a correct gradient. NaNs and infs appear in the gradients where they should not and no method has
been found to eliminate these pesky intrusions.

NOTE: The DTW distances are correct even in batch mode. It's only the gradients which are wrong.

The Theano implementation is slow! While the Cython implementation can compute the DTW distance between 100 sequence
pairs in 0.079 seconds, the nonbatch Theano implementation takes 19.5 seconds and the fully batched Theano
implementation takes 0.5 seconds. Little attempt has been made to optimize either Theano implementation at this point as
the focus has been on getting the gradients to work.

Requirements
------------
Python (tested with 2.7.9)
Theano (tested with 0.7)
speech_dtw (optional; see framework/cython_dtw.py for details)

Also all of Theano's dependecies, esp. numpy.

Getting started
---------------
Run dtw.py. See the comments at the top of that file for more information.

Collaborators
-------------
Daniel Renshaw
Herman Kamper
Sharon Goldwater

License
-------
Copyright 2014 Daniel Renshaw

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
