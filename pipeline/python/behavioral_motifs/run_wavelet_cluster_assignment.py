#
# This script smoothness an embedding space, applies watershed, and subsequently assigns each timepoint a clusterID.
#

import os, glob, numpy as np, pandas as pd, numba as nb, regex as re
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend

from sklearn.neighbors import NearestNeighbors
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from sklearn.mixture import GaussianMixture
from scipy import ndimage as ndi

DEBUG = False
WATERSHED_SINGLELEVEL = True

# ==============================================================================================================
# Watershed / Smoothing
# ==============================================================================================================

@nb.jit(nopython=True, fastmath=True)
def histConvolved(arrDR, bins, kernelStd, kernelScale = 25):
    heatmap = np.zeros(bins, dtype=np.float64)
    for i in range(arrDR.shape[0]):
        vx = kernelStd[i] * bins[0] * kernelScale
        vy = kernelStd[i] * bins[1] * kernelScale
        vx = max(1, vx)
        vy = max(1, vy)
        sx = int(np.floor(vx * 10))
        sy = int(np.ceil (vy * 10))
        cx, cy = arrDR[i,:]
        s = 0
        # Make sure that we normalize the contribution of each timepoint
        for dx in range(-sx, sx+1):
            for dy in range(-sy, sy+1):
                x = int(cx * bins[0]) + dx
                y = int(cy * bins[1]) + dy
                # Note: we make the conscious choice here to not normalize only by pixels within the bounds of the histogram
                s += np.exp(-0.5 * ((dx/vx)**2 + (dy/vy)**2))
        # Draw a gaussian at every timepoint
        for dx in range(-sx, sx+1):
            for dy in range(-sy, sy+1):
                x = int(cx * bins[0]) + dx
                y = int(cy * bins[1]) + dy
                if x >=0 and y>= 0 and x < bins[0] and y < bins[1]:
                    heatmap[x, y] += np.exp(-0.5 * ((dx/vx)**2 + (dy/vy)**2)) / s
    return heatmap

def computeWatershed(fnameEmbeddingSpace, fname, fnameWatershed):
    """
    Compute smoothed versions of this embedding space, together with watershed.

    :param fnameEmbeddingSpace: The parent embedding space (consensus or single-file)
    :return:
    """

    # Find filtered files
    globPattern = str(fname)
    globPattern = re.sub('(?<=behavior/)[a-zA-Z0-9\\-\\_\\~]*', '*', globPattern)
    globPattern = fixLongPath(globPattern)
    fnamesFiltered = glob.glob(globPattern)

    arrFilt = [np.load(x) for x in fnamesFiltered]
    arrFilt_ = [arrFilt[i][np.random.choice(np.argwhere(~np.any(
        np.isnan(arrFilt[i]), axis=1)).T[0], -1, replace=False), 0:2] for i in range(len(arrFilt))]

    arrFiltAll = np.vstack(arrFilt_)

    neigh_dist, neigh_ind = NearestNeighbors(n_neighbors=10).fit(arrFiltAll).kneighbors(arrFiltAll, n_neighbors=10)
    kernelStd = np.mean(neigh_dist, axis=1)

    nbinsX = 200 #np.histogram(arrFiltAll[:, 0], bins='fd')[0].shape[0]
    nbinsY = 200 #np.histogram(arrFiltAll[:, 1], bins='fd')[1].shape[0]

    numHeatmaps = 99
    kernelScales = np.arange(1, 1 + numHeatmaps) if not WATERSHED_SINGLELEVEL else np.ones(1, dtype=int)

    heatmaps = Parallel(n_jobs=1 if (DEBUG or WATERSHED_SINGLELEVEL) else 50)(delayed(histConvolved)(
        arrFiltAll, (nbinsX, nbinsY), kernelStd, kernelScale) for kernelScale in tqdm(kernelScales, disable=(not DEBUG)))

    if WATERSHED_SINGLELEVEL:
        heatmaps = heatmaps * numHeatmaps

    # Save in output array
    arrOut = np.zeros((nbinsX, nbinsY, numHeatmaps, 2), dtype=np.float64)
    arrOut[:, :, :, 0] = np.moveaxis(np.array(heatmaps), 0, 2)

    for i in range(numHeatmaps):
        Hsmooth = arrOut[:, :, i, 0]

        # Obtain local maximima
        local_maxi = peak_local_max(Hsmooth, indices=False, footprint=np.ones((3, 3)))

        # Perform watershed
        markers = ndi.label(local_maxi)[0]
        labels = watershed(-Hsmooth, markers=markers)

        # Store in output array
        arrOut[:, :, i, 1] = labels

    npsave(fnameWatershed, arrOut)

def fixLongPath(x):
    if x.startswith('\\\\?\\'):
        return x.replace('/', '\\')
    else:
        return '\\\\?\\' + x.replace('/', '\\')

def npsave(fname, arr):
    # The '\\?\' prefix overcomes path length limitations...
    np.save(fixLongPath(fname), arr)

def npload(fname):
    # The '\\?\' prefix overcomes path length limitations...
    return np.load(fixLongPath(fname))

# ==============================================================================================================
# Main Function
# ==============================================================================================================

@nb.njit
def _getClusterIDs(arrC, arrWatershed):
    clusterIDs = np.full((arrC.shape[0], arrWatershed.shape[2] + 5), -1, dtype=np.double) # int64
    for j in range(arrWatershed.shape[2]):
        for i in range(arrC.shape[0]):
            if arrC[i, 0] >= 0:
                c1 = max(0, min(arrWatershed.shape[0]-1, arrC[i, 0]))
                c2 = max(0, min(arrWatershed.shape[1]-1, arrC[i, 1]))
                clusterIDs[i, j] = arrWatershed[c1, c2, j, 1]
    return clusterIDs

def assignClusters(fname, overwrite = False):
    fnameClusters = fixLongPath(fname.replace('.filtered2.npy', '').replace('.filtered.npy', '') + '.clusters.npy')

    if os.path.exists(fnameClusters) and not overwrite:
        return

    # Get the filename of the space this was produced from
    if not '/embeddings/' in fname.replace('\\','/'):
        fnameEmbeddingSpace = glob.glob(fixLongPath(os.path.join('Y:/wavelet/**/',
            os.path.basename(fname).replace('.filtered2','').replace('.filtered',''))))
    else:
        fnameEmbeddingSpace = glob.glob(fixLongPath(os.path.join(os.path.dirname(fname), '**',
            os.path.basename(fname).replace('.filtered2','').replace('.filtered',''))))
    fnameEmbeddingSpace = [x for x in fnameEmbeddingSpace if not 'old' in x]

    # Should find only 1 (not more or less) parent embedding files.
    if len(fnameEmbeddingSpace) == 0:
        raise Exception('No embedding file found.')
    elif len(fnameEmbeddingSpace) > 1:
        raise Exception('Too many (>1) embedding file candidates found.')
    else:
        fnameEmbeddingSpace = fnameEmbeddingSpace[0]

    # Check if smoothed / watershed versions of this file have been computed?
    fnameWatershed = fnameEmbeddingSpace.replace('.npy','') + '.smoothed.watershed.npy'
    if not os.path.exists(fnameWatershed):
        computeWatershed(fnameEmbeddingSpace, fname, fnameWatershed)

    # Now that the watershed file exists, load the watershed cluster IDs
    arrWatershed = npload(fnameWatershed)[:,:,0:100,:]

    # Load recording
    arr = npload(fname)
    segmentIDs = arr[:,4].astype(np.int64)
    isPause = np.all(~np.isnan(arr[:,0:2]), axis=1)
    arrOrg = arr.copy()

    #
    arrC = arr[:,2:4].copy()
    arrC[:,0] *= arrWatershed.shape[0]
    arrC[:,1] *= arrWatershed.shape[1]
    arrC = np.round(arrC)
    arrC[:,0] = np.clip(arrC[:,0], 0, arrWatershed.shape[0])
    arrC[:,1] = np.clip(arrC[:,1], 0, arrWatershed.shape[1])
    arrC[np.isnan(arrC)] = -1
    arrC = np.round(arrC).astype(np.int64)

    clusterIDs = _getClusterIDs(arrC, arrWatershed)

    # Save segmentIDs for easy reference
    clusterIDs[:,-5] = arr[:,0]
    clusterIDs[:,-4] = arr[:,1]
    clusterIDs[:,-3] = arrOrg[:,0]
    clusterIDs[:,-2] = arrOrg[:,1]
    clusterIDs[:,-1] = segmentIDs

    # Save clusterIDs
    npsave(fnameClusters, clusterIDs)

    # Done!
    pass

# ==============================================================================================================
# Entry point
# ==============================================================================================================

def findUnprocessedRecordings(dir, overwrite=False):
    pat = os.path.join(dir, '**/rawmvmt*.filtered2.npy').replace('\\', '/')
    fnames = [x.replace('\\', '/') for x in tqdm(glob.iglob(pat, recursive=True), desc='enumerating files')]
    pat = os.path.join(dir, '**/wavelet*.filtered2.npy').replace('\\', '/')
    fnames+= [x.replace('\\','/') for x in tqdm(glob.iglob(pat, recursive=True), desc='enumerating files')]
    fnames = [x for x in fnames if 'tsne' in x or 'umap' in x]
    fnames = [x for x in fnames if 'old' not in x]
    fnames = [x for x in fnames if 'croprot' not in x and re.match('.*/embeddings/wavelet_.*/wavelet_.*', x) is None \
              and ('/wavelet/' in x or '/embeddings/' in x)]
    fnames = [x for x in fnames if not x.endswith('.filtered.filtered.npy')] # Filter out accidentally double-processed files
    fnames = [x for x in fnames if not x.endswith('.filtered2.filtered2.npy')] # Filter out accidentally double-processed files

    if not overwrite:
        fnames = [x for x in fnames if not os.path.exists(x.replace('.npy','') + '.clusters.npy')]

    return fnames

def assignClustersSafe(fn):
    try:
        print('Processing: {}'.format(fn[-200:]))
        assignClusters(fn)
    except Exception as e:
        print(fn, e)

if __name__ == "__main__":

    DEBUG_FILE = None

    if DEBUG_FILE is not None:
        if isinstance(DEBUG_FILE, list):
            if not WATERSHED_SINGLELEVEL:
                for df in tqdm(DEBUG_FILE):
                    print(df)
                    assignClusters(df, overwrite=True)
            else:
                Parallel(n_jobs=1 if DEBUG else 55)(delayed(assignClusters)(df, overwrite=True) for df in tqdm(DEBUG_FILE))
        else:
            assignClusters(DEBUG_FILE, overwrite=True)
    else:
        from pipeline.python.misc import gui_misc

        r = gui_misc.askUser('File or directory?', 'Do you want to process a single file (Yes) or a whole directory (No)?')

        if r:
            selectedFile = gui_misc.askUserForFile()
            assignClusters(selectedFile)
        else:
            selectedDir, OVERWRITE_RECORDING, nRec, fnames = gui_misc.askUserForDirectory(
                findUnprocessedRecordings, returnFilenames=True, forceOverwrite=False)
            print('\n'.join(fnames))
            Parallel(n_jobs=1 if DEBUG else 55)(delayed(assignClustersSafe)(fn) for fn in tqdm(fnames))