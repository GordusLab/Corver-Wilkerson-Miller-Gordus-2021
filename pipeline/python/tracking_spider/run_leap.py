
# =====================================================================================
#
# This script runs LEAP on directories that have already been processed
#
# =====================================================================================

# Settings: Location of LEAP trained model
LEAP_MODEL = 'D:/data/LEAP/10K_no_rot_no_mirror/final_model.h5'

# Settings: Batch size to run LEAP in
LEAP_BATCH_SIZE = 10000

# =====================================================================================
# Imports & Globals
# =====================================================================================

# This fixes GPU memory allocation errors
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Import main libraries
import os, sys, regex as re, itertools, numpy as np

# Import progress bar library
from tqdm import tqdm as tqdm

# Import libraries for high-performance and parallel computing
from joblib import Parallel, delayed

# Add 'libraries' directory to path (to allow LEAP to be imported)
ROOT = os.getcwd()[:os.getcwd().rfind('spider-behavior') +  len('spider-behavior')];
sys.path.append(os.path.join(ROOT,'libraries/leap'))

# Import LEAP
import keras
import keras.models
from leap.predict_box import convert_to_peak_outputs
from leap.utils import versions

# Set process priority to lowest so this script doesn't interfere with OS function
import psutil
p = psutil.Process()
try:
    p.nice(psutil.IDLE_PRIORITY_CLASS)
except:
    p.nice(20)

# =====================================================================================
# Main LEAP-invoking function
# =====================================================================================

def processRecording(fname, overwrite=False):

    if fname is None:
        return # Empty filename, so return

    # Load memory-mapped file & get number of images it contains
    vid = np.memmap(fname, mode='r', dtype=np.uint8)
    N = int(vid.shape[0] / (200 * 200))
    del vid;  # Closes file

    # Re-open with correct dimensions
    vid = np.memmap(fname, mode='r', dtype=np.uint8, shape=(N, 200, 200))

    # Load model
    model = convert_to_peak_outputs(keras.models.load_model(LEAP_MODEL))

    # Create LEAP output
    fnameLeap = fname.replace('_img.npy', '') + '_leap.npy'

    # Open output file (memory-mapped)
    predLeap = np.memmap(fnameLeap, shape=(N, 3, 26), mode='w+', dtype=np.float32)

    for i in tqdm(range(int(np.ceil(N / LEAP_BATCH_SIZE))), desc='Running LEAP'):
        i1 = (i * LEAP_BATCH_SIZE)
        i2 = min(vid.shape[0] - 1, (i + 1) * LEAP_BATCH_SIZE)
        predLeap[i1:i2, :, :] = model.predict(vid[i1:i2, :, :, np.newaxis])

    # Close output file by deleting memory-mapped array
    del predLeap

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
                fnameLeap = fname.replace('_img.npy', '') + '_leap.npy'
                if overwrite or not os.path.exists(fnameLeap):
                    dirs.append(fname)
    return dirs

def isRecordingDir(dirpath):
    return os.path.exists(os.path.join(dirpath, 'raw')) or os.path.exists(os.path.join(dirpath, 'ufmf'))

# =====================================================================================
# Run this script on an entire directory
# =====================================================================================

if __name__ == "__main__":

    # Print available hardware
    print(versions(list_devices=True))

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
            njobs = 1 # Note: Due to GPU memory limitations, only 1 LEAP job can be run simultaneously
            if njobs == 1:
                [processRecording(f, overwrite=OVERWRITE_RECORDING) for f in tqdm(fnames)]
            else:
                Parallel(n_jobs=njobs)(delayed(processRecording)(f, overwrite=OVERWRITE_RECORDING) for f in tqdm(fnames))
