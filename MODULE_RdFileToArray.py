import os, sys, numpy as np


#pre allocating memory would be faster than appending for training data construction

#function reads the training input txt file and outputs to a numpy (np.) array
def input_array_get(countTypeFile):
    
    training_input_list = []

    #locates file in the same path as the executing python file
    fh = open(os.path.join(sys.path[0],'%s' % countTypeFile))

    #assigns fh.readlines functions value to variable "line"
    for line in fh.readlines():
        training_input_list.append([])

        for i in line.split(','):
            training_input_list[-1].append(int(i))

    fh.close()

    return np.array(training_input_list)


def output_array_get(countTypeFile):
    
    training_output_list = []

    #locates file in the same path as the executing python file
    fc = open(os.path.join(sys.path[0],'%s' % countTypeFile))

    #assigns fh.readlines functions value to variable "line"
    for line in fc.readlines():
        training_output_list.append([])

        for i in line.split(','):
            training_output_list[-1].append(int(i))

    fc.close()

    return np.array(training_output_list)

