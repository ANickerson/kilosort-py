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


def batch_process(data, batch_size=1000, parallel=False):
    """
    Proces a signal by batch

    :param data: raw signal to process
    :param batch_size: size of the signal to run
    :param parallel: execute in parallel @TODO: implement parallel batch processing
    """
    pass


def preprocess_data(sig,
                    filtertype="butter",
                    lfreq=300,
                    hfreq=20000,
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
    signalarr = zca_whiten(signalarr)


def zca_whiten(data):
    """zca whitening method for data."""
    m = np.mean(data, 0) # take mean
    _data = data - m # demean data
    cov = np.dot(_data.T, _data) / (_data.shape[0] - 1) #dot product of data.T, data devide by len-1
    U, S, _ = np.linalg.svd(cov) # svd of covariance matrix
    s = np.sqrt(S)  #S.clip(self.regularization)) 
    s_inv = np.diag(1. / s)
    s = np.diag(s)
    _whiten = np.dot(np.dot(U, s_inv), U.T)
    return np.dot(_data, _whiten.T)
    #self.dewhiten_ = np.dot(np.dot(U, s), U.T)


class TheanoFunctions:
    def __init__(self):
        """cached theano functions. Probably not the best way to do this."""
        self._ZCAWhiten = None


    def zca_whiten(self, data):
        """
        zca whitening method for data.
        """
        if self._ZCAWhiten is None:

            x = T.dmatrix('x', T.floatX)
            m_ = x.mean(0)
            data_ = x - m_
            cov = theano.dot(x.T, x) / (x.shape[0] - 1)
            U, S, _ = nlinalg.svd(cov)
            s = S.sqrt()
            s_inv = T.diag(1. / s)
            s = T.diag(s)
            whiten_ = U.dot(s_inv).dot(U.T)
            out = data_.dot(whiten_.T)

            self._ZCAWhiten = theano.function([x], out)
        return self._ZCAWhiten(data)
