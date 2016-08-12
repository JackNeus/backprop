This repository will be home to all code related to the neural net I am working on. 

The math was mostly based off of a journal article written by Rumelhart, Hinton, and Williams, although I used a different squashing function (tanh) to achieve a range of -1 to 1 instead of 0 to 1.

The net is initialized with random values, so consecutive runs of a test may not all yield correctly-trained nets. 
In the Parity test (with N = 5), for example, 1000 training rounds yield a correct net maybe 3 out of 10 times. So with some tests it is possible to achieve a correctly-trained net by re-running the test multiple times.

WORKING EXAMPLES (with included .net files):

XOR (xor.net)

PARITY (parity5.net - N = 5)

TODO:

Research momentum term

Add more examples

Make it easier for others to set up tests
