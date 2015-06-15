# TheanoBatchDTW
A batch (multiple concurrent sequence pairs) implementation of Dynamic Time Warping (DTW) in Theano.

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


Requirements
------------
* Python (tested with 2.7.9)
* Theano (tested with 0.7)
* speech_dtw (optional; see framework/cython_dtw.py for details)

Also all of Theano's dependencies, esp. numpy.

Getting started
---------------
Run dtw.py. See the comments at the top of that file for more information.

Collaborators
-------------
* Daniel Renshaw
* Herman Kamper
* Sharon Goldwater

License
-------
Copyright 2015 Daniel Renshaw

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
