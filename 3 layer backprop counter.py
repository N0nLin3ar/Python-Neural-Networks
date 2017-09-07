#!/usr/bin/env python

#import tensorflow as tf
import numpy as np
from MODULE_RdFileToArray import input_array_get, output_array_get


def nonlin(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

countType = 0

while countType not in ('7', '15'):
    countType = str(raw_input("Would you like to train to count to (7 or 15)? "))

print ('You selected: ' + countType)

inputCountFile = ('%s Count Input.txt' % countType)

print(inputCountFile)

outputCountFile = ('%s Count Output.txt' % countType)

X = input_array_get(inputCountFile)
x1,y1 = X.shape
#print '\n', ("Input Training Data:"), '\n', X


y = output_array_get(outputCountFile)
x2,y2 = y.shape
#print '\n', ("Output Training Data:"), '\n',


#set the number of hidden layers (x3):
x3 = int(y1+3)

np.random.seed(1)

# randomly initialize our weights with mean 0.
# y1 = width of training data. y2 = width of output data. 
# x3 = number of hidden nodes
syn0 = 2*np.random.random((y1,x3)) - 1
syn1 = 2*np.random.random((x3,y2)) - 1

print '\n', ("INITIAL WEIGHTS:"), '\n', syn0, '\n', '\n', syn1, '\n', '\n'

#print '\n',("Original Randomized Network:"),'\n', syn0, '\n','\n', syn1, '\n'

for j in xrange(80000):

	# Feed forward through layers 0, 1, and 2
	l0 = X
	l1 = nonlin(np.dot(l0,syn0))
	l2 = nonlin(np.dot(l1,syn1))

	# how much did we miss the target value?
	l2_error = y - l2

	if (j% 10000) == 0:
		print "L2 training accuracy:  " + str(1 - (float(np.mean(np.abs(l2_error)))))

	# in what direction is the target value?
	# were we really sure? if so, don't change too much.
	l2_delta = l2_error*nonlin(l2,deriv=True)

	# how much did each l1 value contribute to the l2 error (according to the weights)?
	l1_error = l2_delta.dot(syn1.T)

	# in what direction is the target l1?
	# were we really sure? if so, don't change too much.
	l1_delta = l1_error * nonlin(l1,deriv=True)

	syn1 += l1.T.dot(l2_delta)
	syn0 += l0.T.dot(l1_delta)

print '\n', '\n', ("TRAINED NETWORK:"),'\n', syn0, '\n','\n', syn1


# This allows for manually testing the network
while raw_input("Would you like to test the network (y / n)?  ") == 'y':

	t0 = []
	while len(t0) < y1:
		testInput = int(raw_input("Enter one unit of binary test data (0 or 1), then 'enter':  "))
		t0.append(testInput)
		print ("Test data recorded so far:  "), t0

	t0 = np.array(t0)
	t1 = nonlin(np.dot(t0,syn0))
	t2 = nonlin(np.dot(t1,syn1))
	print '\n', ("The raw answer is:  "), t2

	print '\n', ("The rounded answer is:  "), np.around(t2, decimals = 0)

#    else:
#        break #stops the loop, ends the program
