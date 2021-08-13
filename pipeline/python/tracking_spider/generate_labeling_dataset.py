
"""
    This script extracts frames to be labeled from the cropped/rotated spider videos, and saves them in a format
    that can be loaded and manually annotated using LEAP's Matlab 'annotate_joints' GUI.
"""

# =====================================================================================
# Imports & Globals
# =====================================================================================

# Import main libraries
import scipy.io, h5py, os, numpy as np, shutil
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.ndimage
from tqdm import tqdm as tqdm

# Find root directory, in order to locate MAT template...
ROOT = os.getcwd()[:os.getcwd().rfind('spider-behavior') +  len('spider-behavior')];

# Determine location of the template
# Note: This template was originally generated using LEAP's Matlab GUI, and has been re-exported
#       to this ver. 7.3 MAT file, which is an H5 file format.
LABELS_MAT_TEMPLATE = os.path.join(ROOT, 'pipeline/models/label_joints/label_data.labels.mat')

if not os.path.exists(LABELS_MAT_TEMPLATE):
    raise Exception("Could not locate MAT label template!")

# Set process priority to lowest so this script doesn't interfere with OS function
import psutil
p = psutil.Process()
try:
    p.nice(psutil.IDLE_PRIORITY_CLASS)
except:
    p.nice(20)

# =====================================================================================
# This function selects spider images based on KMeans clustering
# =====================================================================================

def selectFramesKMeans(fname, arr, Nimages, maxFramesToConsider = 10000):
    # Determine filename containing positions
    # Note: We (arbitrarily) try LEAP-determined positions first, then DLC if LEAP file doesn't exist
    fnamePos = fname.replace('_img.npy', '_leap_position_orientation.npy')
    if not os.path.exists(fnamePos):
        fnamePos = fname.replace('_img.npy', '_dlc_position_orientation.npy')
    if not os.path.exists(fnamePos):
        print("Position/orientation has not been computed by either LEAP or DLC, cannot proceed with generating labeling dataset. Skipping...")
        return None
    # Compute velocity (and use erosion (2x) to remove outlier movement frames)
    # Note: We use 0.25 pixels/frame as velocity threshold. This corresponds with approximately 1 mm / second.
    xy = np.load(fnamePos)[:, [0, 1]]
    vel = np.linalg.norm(np.pad(np.diff(xy, axis=0), (1, 0), mode='constant'), axis=1)
    indices = scipy.ndimage.binary_erosion(vel > 0.25, iterations=2)
    # Determine indices to use
    indices = np.random.choice(np.argwhere(indices).T[0], maxFramesToConsider)
    # Flatten image
    arrFlat = arr.reshape((arr.shape[0], 200 * 200))[indices, :]
    # Reduce images
    _pca = PCA(n_components = 32)
    _pca.fit(arrFlat)
    # Check percentage variance explained (could print this, for now we don't)
    np.sum(_pca.explained_variance_ratio_)
    # Reduce images
    _arrPCA = _pca.transform(arrFlat)
    # Cluster in PCA space
    _kmeans = KMeans(n_clusters = Nimages)
    _kmeans.fit(_arrPCA)
    # Now choose one image from each cluster
    idxImg = []
    for cl in np.unique(_kmeans.labels_):
        idxImg.append( np.random.choice(indices[_kmeans.labels_ == cl]) )
    # If array is too small, sample again from these clusters
    cl = 0
    while len(idxImg) < Nimages:
        idx = None
        tries = 0
        while idx is None or idx in idxImg:
            if np.sum(_kmeans.labels_ == cl) == 0:
                break
            idx = np.random.choice(indices[_kmeans.labels_ == cl])
            tries += 1
            if tries > 1000:
                break
        if idx is not None:
            idxImg.append(idx)
        cl = (cl + 1) % np.max(_kmeans.labels_)
    # Return indices in random order
    np.random.shuffle(idxImg)
    # Done!
    return idxImg

# =====================================================================================
# Main function
# =====================================================================================

def processRecording(fname, overwrite=False):

    if fname is None:
        return # Empty filename, so return

    print('Processing {}'.format(fname[-200:]))

    # Number of images to manually annotate
    Nannotate = 500

    # Load images
    arr = np.memmap(fname, dtype=np.uint8, mode='r')
    N = int(arr.shape[0] / (200 * 200))
    assert arr.shape[0] % (200 * 200) == 0
    del arr
    arr = np.memmap(fname, dtype=np.uint8, mode='r', shape=(N, 200, 200))

    # Select frames to label
    idxImg = selectFramesKMeans(fname, arr, Nannotate, maxFramesToConsider=10000)
    if idxImg is None:
        return # Something went wrong (e.g. position not available), so return...

    # Save indices used
    np.save         (fname.replace('_img.npy', '') + '.manual_annotation.frames_used.npy', idxImg)
    scipy.io.savemat(fname.replace('_img.npy', '') + '.manual_annotation.frames_used.mat', {'frames_used': idxImg})

    # Generate H5 file
    fNew = h5py.File(fname.replace('_img.npy', '') + '.manual_annotation.h5', 'w')
    ds = fNew.create_dataset("box", (Nannotate, 1, 200, 200), dtype='<f8')
    for i in range(Nannotate):
        try:
            ds[i,0,:,:] = arr[idxImg[i],:,:].T # The image needs to be transposed to appear in the correct orientation
        except Exception as e:
            print(e)
            print(ds.shape, arr.shape, i, len(idxImg))
    fNew.close()

    # Copy template file into new filename
    fnameMat = fname.replace('_img.npy', '') + '.manual_annotation.labels.mat'
    shutil.copy(LABELS_MAT_TEMPLATE, fnameMat)

    # Open and resize
    matNew = h5py.File(fnameMat, 'a')

    # Resize label data array
    #   Note: Setting to NaN is required to indicate this label is not set
    del matNew['initialization']
    matNew.create_dataset("initialization", (Nannotate, 2, 26), '<f4')
    matNew['initialization'][:] = np.nan

    # Resize label data array
    #   Note: Setting to NaN is required to indicate this label is not set
    del matNew['positions']
    matNew.create_dataset("positions", (Nannotate, 2, 26), '<f4')
    matNew['positions'][:] = np.nan

    # Make sure we start at the first frame (avoids out-of-bounds errors)
    matNew['config/initialFrame'][:] = 1

    matNew.close()

    # Done!

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
                fnameLabels = fname.replace('_img.npy', '') + '.manual_annotation.labels.mat'
                if overwrite or not os.path.exists(fnameLabels):
                    dirs.append(fname)

    print('Discovered {} recording directories to be processed.'.format(len(dirs)))

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
            njobs = 1
            if njobs == 1:
                [processRecording(f, overwrite=OVERWRITE_RECORDING) for f in tqdm(fnames)]
            else:
                Parallel(n_jobs=njobs)(delayed(processRecording)(f, overwrite=OVERWRITE_RECORDING) for f in tqdm(fnames))
