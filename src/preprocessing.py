"""
Preprocess the recordings.

Read format

butterworth filter 

whitening

"""

import numpy as np
import theano
from theano import tensor as T
from theano.tensor import nlinalg
from scipy import signal
from algorithms import alg

def batch_process(data, batch_size=1000, parallel=False):
    """
    Proces a signal by batch

    :param data: raw signal to process
    :param batch_size: size of the signal to run
    :param parallel: execute in parallel @TODO: implement parallel batch processing
    """
    pass


def preprocess_data(sig,
                   # filtertype="butter",
                    lfreq=300,
                   # hfreq=20000,
                    whiten=True):
    """
    run the process steps on the given input data segment.

    :type sig: config.signal
    """

    signalarr = sig.data
    #highpass filter at 300Hz
    a, b = signal.butter(3, (lfreq * 2) / (sig.fs), "highpass")
    signalarr = signal.filtfilt(signalarr, a, b) #@TODO: implement filter in theano?

    # common average reference
    signalarr -= np.median(signalarr, 0)[:, np.newaxis]

    #whiten data to remove correlated noise
    #and we therefore compute the columns of the whitening matrix W ZCA
    # independently for each channel,  based on its nearest 32 neighbors.
    # @TODO: sort this out!
    for x in range(signalarr.shape[0]):
        #get 32 nearest channels
        if x<16:
            x0 = 0
            x1 = 32
        else:
            x0 = x -16
            x1 = x + 16
        x1  = x0 + 32
        if x1>signalarr.shape[0]:
            x0 -= (x1 - signalarr.shape[0])
            x1 = signalarr.shape[0] 
        signalarr2 = alg.zca_whiten(signalarr[x0:x1,:])
        signalarr[x] = signalarr2[x-x0]

    # Return the data (cleanedsignal,)
    return signalarr