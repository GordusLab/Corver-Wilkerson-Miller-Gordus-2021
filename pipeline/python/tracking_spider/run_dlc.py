
"""
    This script runs DeepLabCut (DLC) on directories that have been processed using the
    crop/rotate script.
"""

# Settings: Location of DLC trained model
#     The model directories to attempt to locate, in order...
#     By providing multiple directories, this notebook can be run on multiple workstations
MODEL_DIRECTORIES = [
    'Z:/behavior/DLC/leg_tracking-Nick-2019-03-01/',
    'B:/Spider Code/dlc/leg_tracking-Nick-2019-03-01/'
]

# =====================================================================================
# Imports & Globals
# =====================================================================================

# This fixes GPU memory allocation errors
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Import main libraries
import os, sys, regex as re, itertools, numpy as np, pandas as pd

# Import progress bar library
from tqdm import tqdm as tqdm

# Import libraries for high-performance and parallel computing
from joblib import Parallel, delayed

# Add 'libraries' directory to path (to allow DLC to be imported)
ROOT = os.getcwd()[:os.getcwd().rfind('spider-behavior') +  len('spider-behavior')];
sys.path.append(os.path.join(ROOT,'libraries/DeepLabCut'))

# Import DLC
import deeplabcut

# Set process priority to lowest so this script doesn't interfere with OS function
import psutil
p = psutil.Process()
try:
    p.nice(psutil.IDLE_PRIORITY_CLASS)
except:
    p.nice(20)

# Detect directory
MODEL_DIRECTORY = None
for d in MODEL_DIRECTORIES:
    if os.path.exists(d):
        MODEL_DIRECTORY = d
        break
MODEL_DIRECTORY

# =====================================================================================
# Reader class for our cropped/rotated videos, which helps DLC reads these videos.
# =====================================================================================

class SpiderCroppedVideoReader():
    """ This class mimics OpenCV's VideoCapture class, but instead reads
        the *_img.npy files, which contain 200x200 cropped/rotated and monochrome
        spider images.
    """

    def __init__(self, filename):
        self.fname = filename
        self.iframe = 0

        # DEBUG
        self.MAX_NUM_FRAMES = 50 * 3600 * 24 * 100

        # Is this a supported input file?
        if not self.fname.endswith('_img.npy'):
            raise Exception('Unsupported input file. Only *_img.npy files allowed.')

        # Open file to determine size
        import numpy as np
        self.arr = np.memmap(self.fname, dtype=np.uint8, mode='r')
        N = self.arr.shape[0]
        del self.arr
        # Ensure this file has the right size
        if N % (200 * 200) != 0:
            raise Exception('Image does not have expected size of 200x200.')
        # Open again with right shape
        self.arr = np.memmap(self.fname, dtype=np.uint8, mode='r', shape=(int(N / (200 * 200)), 200, 200))

    # Get various metadata
    def get(self, i):
        # NUM. FRAMES
        if i == 7:
            if self.arr is not None:
                return min(self.MAX_NUM_FRAMES, self.arr.shape[0])
            else:
                raise Exception('File closed.')
        # FPS
        elif i == 5:
            return 50
        # HEIGHT
        elif i == 4:
            return 200
        # WIDTH
        elif i == 3:
            return 200
        # ERROR
        else:
            raise Exception('Unsupported metadata requested: {}'.format(i))

    # Return if this file is opened
    def isOpened(self):
        return (self.arr is not None)

    # Return the next frame
    def read(self):
        if self.iframe >= min(self.MAX_NUM_FRAMES, self.arr.shape[0]):
            del self.arr
            self.arr = None
            return False, None
        else:
            f = self.arr[self.iframe, :, :]
            # Convert mono to RGB so DLC can process it correctly
            f = np.repeat(f[:, :, np.newaxis], 3, axis=2)
            self.iframe += 1
            return True, f

# =====================================================================================
# Main DLC-invoking function
# =====================================================================================

def processRecording(fname, overwrite=False):

    if fname is None:
        return # Empty filename, so return

    # Get DLC config file
    fnameConfig = os.path.join(MODEL_DIRECTORY, 'config.yaml')

    # Run DLC, and use the custom video reader class
    deeplabcut.analyze_videos(
        fnameConfig, [fname, ],
        shuffle=1, save_as_csv=False, videotype='.npy', videoReader=SpiderCroppedVideoReader, overwrite=overwrite)

    # Now reformat the DLC output into a raw NumPy array, with the exact same format of the
    # LEAP output array, thereby making these files interchangeable.

    # Now find the DLC output file
    dlcFiles = [os.path.join(os.path.dirname(fname), f) for f in \
                os.listdir(os.path.dirname(fname)) if \
                os.path.basename(fname).replace('.npy', 'DeepCut') in f]

    dlcH5 = [x for x in dlcFiles if x.endswith('.h5')][0]

    # Read DLC output
    d = pd.read_hdf(dlcH5, 'df_with_missing')

    # Use same dataframe ordering as LEAP
    VARIABLE_ORDER = ['Prosoma', 'Midpoint',
                      'R1_base', 'R2_base', 'R3_base', 'R4_base',
                      'R1_femur', 'R2_femur', 'R3_femur', 'R4_femur',
                      'R1_tibia', 'R2_tibia', 'R3_tibia', 'R4_tibia',
                      'L1_base', 'L2_base', 'L3_base', 'L4_base',
                      'L1_femur', 'L2_femur', 'L3_femur', 'L4_femur',
                      'L1_tibia', 'L2_tibia', 'L3_tibia', 'L4_tibia']

    # Determine output filename
    fnameDLC = fname.replace('_img.npy', '') + '_dlc.npy'

    # Open output file (memory-mapped)
    predDLC = np.memmap(fnameDLC, shape=(d.shape[0], 3, 26), mode='w+', dtype=np.float32)

    # Linearize the (currently hierarchical) column names
    d.columns = ['{} {}'.format(*x) for x in zip(d.columns.get_level_values(1), d.columns.get_level_values(2))]
    # Convert the CSV file into the same 3D matrix format as the LEAP data
    for ax in ['x', 'y']:
        for i in range(26):
            predDLC[:, 0 if ax == 'x' else 1, i] = d['{} {}'.format(VARIABLE_ORDER[i], ax)]

    # Close output file by deleting memory-mapped array
    del predDLC

# =====================================================================================
# Helper functions
# =====================================================================================

def getRecordingFilename(d):
    fdir = os.path.join(d, 'croprot/')

    if not os.path.exists(fdir):
        return None

    fname = [os.path.join(fdir, x) for x in os.listdir(fdir) if x.endswith('_img.npy')]
    fname = None if len(fname) != 1 else fname[0]
    return fname

def findUnprocessedRecordings(rootpath, overwrite=True):
    dirs = []
    for x in os.listdir(rootpath) + ['',]:
        fx = os.path.join(rootpath, x)
        # Only proceed if this is a recording directory
        if isRecordingDir(fx):
            # Obtain cropped/rotated filename
            fname = getRecordingFilename(fx)
            if fname is not None:
                fnameDLC = fname.replace('_img.npy', '') + '_dlc.npy'
                if overwrite or not os.path.exists(fnameDLC):
                    dirs.append(fname)
    return dirs

def isRecordingDir(dirpath):
    return os.path.exists(os.path.join(dirpath, 'raw')) or os.path.exists(os.path.join(dirpath, 'ufmf'))

# =====================================================================================
# Run this script on an entire directory
# =====================================================================================

if __name__ == "__main__":
    from pipeline.python.misc import gui_misc
    selectedDir, OVERWRITE_RECORDING, nRec = gui_misc.askUserForDirectory(findUnprocessedRecordings)

    if nRec > 0:
        # Is this a recording dir?
        if isRecordingDir(selectedDir):
            # Process this directory
            fname = getRecordingFilename(selectedDir)
            processRecording(fname, overwrite=OVERWRITE_RECORDING)
        else:
            # Otherwise, this is a root directory with multiple behavior folders...
            # Automatically find the unprocessed behaviors
            fnames = findUnprocessedRecordings(selectedDir, overwrite=OVERWRITE_RECORDING)
            print('\n'.join(fnames))
            njobs = 1 # Note: Due to GPU memory limitations, only 1 LEAP job can be run simultaneously
            if njobs == 1:
                [processRecording(f, overwrite=OVERWRITE_RECORDING) for f in tqdm(fnames)]
            else:
                Parallel(n_jobs=njobs)(delayed(processRecording)(f, overwrite=OVERWRITE_RECORDING) for f in tqdm(fnames))
