"""
Module containing all the individual algorithms. 
It would be possible to use alternative implementations
"""
import numpy as np
import theano
from theano import tensor as T
from theano.tensor import nlinalg

class algorithms_numpy():
    """
    The algorithms implemented in numpy. 
    This should be the base class for any extended implementations

    global notes:
    gather_try is a concept of executing planned matrix operations. Not implmented

    global todos:
    TODO: remove/alter ops. 
    TODO: better commenting of code
    TODO: test output of functions with matlab code - find some input!
    TODO: min function in matlab returns index as well as value. need to fix all calls as the argmin
    # does not work in the same manner. 

    """
    def zca_whiten(self, data):
        """
        zca_whiten the data
        
        TODO: test with oct2py
        """
        m = np.mean(data, 0) # take mean
        _data = data - m # demean data
        cov = np.dot(_data.T, _data) / (_data.shape[0] - 1) #dot product of data.T, data devide by len-1
        U, S, _ = np.linalg.svd(cov) # svd of covariance matrix
        s = np.sqrt(S)  #S.clip(self.regularization)) 
        s_inv = np.diag(1. / s)
        s = np.diag(s)
        _whiten = np.dot(np.dot(U, s_inv), U.T)
        return np.dot(_data, _whiten.T)


    #Main Loop
    def alignW(self, W, ops):
        """
        :param: W: 2d array: nt0, nFilt

        TODO: test using oct2py?
        TODO: find out the use of this function...
        """
        nt0, nFilt = W.shape
        imax = np.argmin(W, axis=0)

        dmax = -(imax - ops.nt0min)
        for i in range(nFilt):
            if dmax[i]>0:
                W[(dmax[i] + 1):nt0, i] = W[1:nt0-dmax[i], i]
            else:
                W[0:nt0+dmax[i], i] = W[(1-dmax[i]):nt0, 1]
        return W
    
    def alignWU(self, WU, ops):
        """
        :param: WU: 3d array: nt0, nChan,nFilt

        TODO: find out what this does
        TODO: test using oct2py 
        """
        nt0, n_chan, n_filt = WU.shape
        imin = np.argmin(WU.reshape(nt0*n_chan, n_filt), axis=0)
        imin_chan = np.ceil(imin/nt0)
        dmax = np.zeros((n_filt,nt0))

        for i in range(n_filt):
            wu = WU[:, imin_chan[i], i]
            imin = np.argmin(wu)
            dmax[i] = -(imin - ops.nt0min)
            if dmax[i]>0:
                WU[(dmax[i] + 1): nt0, :, i] = WU[:nt0-dmax[i],:,1]
            else:
                WU[:nt0+dmax[i],:,i] = WU[(1-dmax[i]):nt0,:,i]
            
        return WU

    def decompose_dWU(self, ops, dWU, n_rank, kcoords):
        """
        :param: dWU: 3d array nt0, n_rank, n_filt

        TODO: find out what this does
        TODO: test using oct2py depends on get_svds and zero_out_kcoords
        """
        nt0, n_chan, n_filt = dWU.shape

        W = np.zeros((nt0, n_rank, n_filt)) #single precision in original code?
        U = np.zeros((n_chan, n_rank, n_filt)) #single precision in orignal code?
        mu = np.zeros((n_filt, 1)) #single precision in orignal code?

        dWU[dWU == np.nan0] = 0 #replace nans
        
        # original code parallel processing option
        # TODO: add parallel processing
        for k in range(n_filt):
            a, b, c = self.get_svds(dWU[:, :, k], n_rank)
            W[:, :, k] = a
            U[:, :, k] = b
            mu[k] = c
        U = np.transpose(U, [1,3,2]) # TODO: improve this?
        W = np.transpose(W, [1,3,2])

        U[U == np.nan] = 0 #replace nans

        if len(np.unique(kcoords)[0]) > 0:
            U = self.zero_out_K_coords(U, kcoords, ops.criterionNoiseChannels)
        
        UtU = np.abs(U[:,:,1].T * U[:,:,1]) > 0.1

        # TODO: change. This seems like a strange function
        Wdiff = np.concatenate((W, np.zeros(2, n_filt,n_rank)), 0) - np.concatenate((np.zeros(2, n_filt, n_rank), W), axis=0)
        
        nu = np.sum( np.sum(Wdiff ** 2, axis=1), axis=3)

        return (W, U, mu, UtU, nu)

    def get_svd(self, dWU, n_rank):
        """
        :param dWU: array to apply svd to.
        TODO: find out what this function does
        TODO: test using oct2py
        """
        Wall, Sv, Uall = np.linalg.svd(dWU) #gather_try?
        imax = np.argmax(np.abs(Wall[:,1]))
        def sign(x):
            x[x > 0] = 1
            x[x < 0] = -1
            return x
        Uall[:,0] = - Uall[:, 0] * sign(Wall[imax, 0])
        Wall[:,0] = - Wall[:, 0] * sign(Wall[imax, 0])

        Wall = Wall * Sv

        Sv = np.diag(Sv)
        mu = np.sum(Sv[1:n_rank] ** 2) ** 0.5
        Wall = Wall/mu
        W = Wall[:, 0:n_rank]
        U = Uall[:, 0:n_rank]
        return (W, U, mu)
    

    def merge_spikes_in(self, uBase, nS, uS, crit):
        """
        TODO: find out what this function does
        check if spikes already in uBase?
        nS is a histogram of some description?
        crit is a criteria for exclusion (similarity?)
        TODO: test using oct2py
        """
        if uBase is None:
            # if uBase is empty then return all the positions
            return ([], np.arange(uS.shape[1]))
        
        cdot = uBase[:,:,0].T * uS[:,:,0]
        for j in range(1,uBase.shape[2]):
            cdot = cdot + uBase[:,:,j].T * uS[:,:,j]
        
        base_norms = np.sum(np.sum(uBase**2, axis=2), axis=0)
        new_norms = np.sum(np.sum(uS**2, axis=2), axis=0)

        c_norms = 1e-10 + np.tile(baseNorms.T, (1, len(new_norms))) \
                         + tile(new_norms, (len(base_norms), 1))
        
        cdot = 1 - 2*(cdot/c_norms)

        imin = np.argmin(cdot, axis=0)
        cdotmin = cdot[imin]

        i_match = cdotmin < crit
        
        nS_new = np.histogram(imin[i_match], np.arange(0, uBase.shape[1])) #not sure this will work
        nS = nS = nS_new

        i_non_match = np.where(cdotmin > crit)

        return (nS, i_non_match)
    
    def mexMPregMUcpu(self, Params, data_raw, fW, data, UtU, mu, lam, dWU, nu, ops):
        """
        I believe this function does the heavy lifting. When using theano this is probably
        the one to reimplement
         get spike times and coefficients

        :params: Params: [NT, n_filt, Th, , , , , pm]
        TODO: figure out what this function does
        TODO: test with oct2py

        TODO: rename
        TODO: change call signature to make more pythonic
        TODO: use a data structure for the raw data
        """
        nt0 = ops.nt0
        NT, n_Filt, Th = Params[0:2]
        pm = Params[8]

        fft_data = np.fft.fft(data,axis=0)
        proj = np.fft.ifft(fft_data * fW[:,:]).real #convolution
        proj = np.sum(proj.reshape(NT, n_filt,3), 2)

        Ci = proj + (mu * lam).T
        Ci = (Ci**2) / (1 + lam.T)
        Ci = Ci - (lam*mu**2).T

        imax = np.argmax(Ci, axis=1)
        mX = Ci[imax]
        maX = -my_min(-mX,31,1) # Err... my_min? This function seems odd. 
        #TODO: convert my_min. or remove?

        st = np.where((maX < mX + 1e-3) & (mX>Th**2))
        st[st>(NT-nt0)]

        imax = imax[st]
        x = []
        cost = []
        nsp = []
        if len(imax)>0:
            inds = st.T + np.arange(nt0).T
            dspk = dataRaw[inds,:].reshape(nt0, len(st), ops.n_chan)
            dspk = np.transpose(dspk, [0, 2, 1])

            x = np.zeros(len(id))
            cost = np.zeros(len(id))
            nsp = np.zeros((n_filt, 1))
            for j in range(dspk.shape[2]):
                dWU[:, :, imax[j]] = pm * dWU[:, :, imin[j]] + (1 - pm) * dspk[:, :, j]
                x[j] = proj[st[j], imin[j]]
                cost[j] = maX[st[j]]
                nsp[imin[j]] = nsp[imin[j]] + 1

            imin = imin - 1
        
        return (dWU, st, id, x, cost, nsp)
    
    def reduce_clusters(self, uS, crit):
        """
        :param uS: 3d array
        TODO: work out what this function does
        TODO: test using matlab/oct2py
        /mainLoop/reduce_clusters.m
        """
        cdot = uS[:, :, 0].T * uS[:, :, 0]
        for j in range(us.shape[2]):
            cdot += uS[:, :, j].T * uS[:, :, j]
        
        # compute norms of each spike
        newNorms = np.sum(np.sum(uS**2, 3), 0)

        # compute sum of pairs of norms
        cNorms = 1e-10 + np.tile(newNorms.T, (1, len(newNorms))) \
                        + np.tile(newNorms, (len(newNorms),1))
        
        # compute normalized distance between spikes
        cdot = 1 - 2 * cdot/cNorms
        cdot = cdot + np.diag(np.inf * np.diag(cdot)) # TODO: what??

        newind = np.argmin(cdot>crit)
        minVal = cdot[newind]


    def update_params(self):
        pass

    def zero_out_K_coords(self, U, kcoords, noise_channels):
        pass

    def initialize_waves0(self):
        pass
    
    def optimize_peaks(self):
        pass
        

class algorithms_theano(algorithms_numpy):
    def __init__(self):
        self._zca_whiten = None

    def zca_whiten(self, data):
        if self._zca_whiten is None:
            x = T.dmatrix(theano.config.floatX)
            x_mean = x.mean(0)
            demeaned_data = x - x_mean
            cov = theano.dot(demeaned_data.T, demeaned_data) / (demeaned_data.shape[0] - 1)
            U, S, _ = nlinalg.svd(cov)
            s = S.sqrt()
            s_inv = T.diag(1. / s)
            s = T.diag(s)
            whiten_ = U.dot(s_inv).dot(U.T)
            out = demeaned_data.dot(whiten_.T)
            self._zca_whiten = theano.function([x], out)
        
        return self._zca_whiten(data)

import tensorflow as tf
class algorithms_tensorflow(algorithms_numpy):
    def __init__(self):
        self._zca_whiten = None
        self.session = tf.Session()

    def zca_whiten(self, data):
        x = tf.placeholder(tf.float32, name="input")
        x_mean = tf.reduce_mean(x,0)
        demeaned_data = x - x_mean
        cov = tf.tensordot(tf.transpose(demeaned_data), demeaned_data,1) / tf.to_float(tf.shape(demeaned_data)[0] - 1)
        U, S, _ = tf.svd(cov)
        s = tf.sqrt(S)
        s_inv = tf.diag(1. / s)
        s = tf.diag(s)
        whiten_ = tf.tensordot(tf.tensordot(U, s_inv,1), tf.transpose(U))
        out = tf.tensordot(demeaned_data, tf.transpose(whiten_),1)

               
        return self.session.run(out, feed_dict={x:data})




alg = algorithms_theano()