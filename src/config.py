"""
Config class for kilsort.


"""


class Config:
    def __init__(self):
        """The configuration class for kilsort.
        @TODO: convert this class into a dictionary.
        """
        self.gpu = True  #This is redundant as it will be handled by theano
        self.parfor = False  #Not sure if this is valid
        self.verbose = True
        self.showfigures = True
        self.datatype = 'dat'  #not sure if it makes sense to give the option for openephys

        self.n_filt = 64  #number of clusters to use
        self.whitening_range = 32  #number of channels to whiten together (-1 for all)

        self.n_rank = 3  #matrix rank of spike template model
        self.nfullpasses = 6  # number of complete passes during optimization
        self.max_fr = 20000  # max no. of spikes to extract per batch

        self.fshigh = 300  # frequency for high pass filtering (Hz)
        self.fslow = 2000  # frequency for low pas filtering (Hz)
        self.ntbuff = 64  # samples of symetrical buffer for whitening and spike detection
        self.scaleproc = 200  #int16 scaling of whitened data :TODO What???
        self.batch_size = 32 * 1024 + self.ntbuff  #batch size

        self.spike_threshold = [
            4, 10, 10
        ]  # threshold for detecting spikes on template-filtered data
        self.large_means_amplitudes = [
            5, 20, 20
        ]  # large means amplitudes are foced around the mean :TODO what?
        self.n_anneal_passes = 4  # should be less than fullpasses :TODO what?
        self.shuffle_clusters = 1  # allow merges and splits during optimisation
        self.merge_threshold = 0.1  # upper threshold for merging
        self.split_threshold = 0.1  # lower threshold for splitting

        #found in preprocessing
        self.nt0 = 61 # Referenced in preprocessing, unsure of use
        self.wPCA = None
        self.fbinary = "" # Filename of 2d electrode array
        self.fproc = "" #not sure, opened with write access?
        self.initialize = "fromData" #initializing function?
        self.chanMap = None # a matlab file for channel mapping or the array itself. 
        # appears to load "chamMap" and "connected" -> a binary array of the channels that are connected?

class Signal:
    def __init__(self, data, fs, coordinatemap, n_channel):
        """A basic class for a signal from a single recording site
        """
        self.data = data
        self.framerate = fs
        self.coords = coordinatemap
        self.n_channel = n_channel  # total number of channels
