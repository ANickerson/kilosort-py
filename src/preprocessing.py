"""
Preprocess the recordings.

Read format

butterworth filter 

whitening

"""
from scipy import signal
import numpy as np


def batchProcess(data, batch_size=1000, parallel=False):
    """
    Proces a signal by batch

    :param data: raw signal to process
    :param batch_size: size of the signal to run
    :param parallel: execute in parallel @TODO: implement parallel batch processing
    """

    
def preprocessData(sig, filtertype="butter", lfreq=300, hfreq=20000, whiten=True):
    """
    run the process steps on the given input data segment.
    
    :type sig: config.signal
    """
   
   signalarr = sig.data 
   #highpass filter at 300Hz
   a, b = signal.butter(3, (lfreq * 2) / (sig.fs), "highpass")
   signalarr = signal.filtfilt(data,a,b)

   # common average reference
   signalarr -= np.median(signalarr,0)[:,np.newaxis]
   

   #whiten data to remove correlated noise
   #and we therefore compute the columns of the whitening matrix W ZCA  independently for each channel,  based on its nearest 32 neighbors. @TODO: sort this out!
   signalarr = ZCAWhiten(signalarr)


def ZCAWhiten(data):
    m_ = np.mean(data, 0)
    data_ = data - m
    cov = np.dot(data_.T, data_) / (data_.shape[0]-1)
    U, S, _ = np.linalg.svd(cov)
    s = np.sqrt(S)#S.clip(self.regularization))
    s_inv = np.diag(1./s)
    s = np.diag(s)
    whiten_ = np.dot(np.dot(U, s_inv), U.T)
    return np.dot(data_, whiten_.T)
    #self.dewhiten_ = np.dot(np.dot(U, s), U.T)


from theano import tensor as T
import theano
from theano.tensor import nlinalg

class theanoFunctions:
    def __init__():
        pass

    def ZCAWhiten(data):
        if self._ZCAWhiten is None:

            x = T.dmatrix('x', theano.floatX)
            m_ = x.mean(0)
            data_ = x - m_
            cov = theano.dot(x.T, x) / (x.shape[0] - 1)
            U, S, _ = nlinalg.svd(cov)
            s = T.sqrt(S)
            s_inv = T.diag(1./s)
            s = T.diag(s)
            whiten_ = T.dot(T.dot(U, s_inv), U.T)
            out = T.dot(data_, whiten_.T)

            self._ZCAWhiten = theano.function([x],out)
        return self._ZCAWhiten(data)

