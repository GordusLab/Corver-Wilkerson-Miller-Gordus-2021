import os, glob, numpy as np, pandas as pd, warnings
from sklearn.mixture import GaussianMixture
from tqdm import tqdm as tqdm
from joblib import Parallel, delayed

def fixLongPath(x):
    if x.startswith('\\\\?\\'):
        return x.replace('/', '\\')
    else:
        return '\\\\?\\' + x.replace('/', '\\')

def filterFile_2(fname, arrDR):
    try:
        # ...
        import miniball
        def computeMiniball(arrDR, i1, i2, window=25):
            windowRad = int(window//2)
            if i2 < 0:
                i2 = arrDR.shape[0]
            return np.array([np.sqrt(miniball.Miniball(arrDR[max(0, i - windowRad):(i + windowRad), 0:2]
                                                       ).squared_radius()) for i in range(i1, i2)])
        radii = computeMiniball(arrDR, 0, -1, 16)
        isPause = (radii < 0.025) # Arbitrary but reasonable cutoff

        # Compute segmentIDs
        segmentIDs = np.full(isPause.shape[0], -1, dtype=np.int64)
        for i in range(isPause.shape[0]):
            if isPause[i]:
                if i == 0:
                    segmentIDs[i] = 0
                elif isPause[i - 1]:
                    segmentIDs[i] = segmentIDs[i - 1]
                else:
                    segmentIDs[i] = np.max(segmentIDs[0:i]) + 1

        # Save the result as an N x 4 array
        arrDRpause = arrDR.copy()
        arrDRpause[~isPause] = np.nan
        arrDRjoint = np.hstack((arrDRpause, arrDR, segmentIDs[:, np.newaxis]))
        fnameOut = fname.replace('.npy', '') + '.filtered2.npy'
        np.save(fixLongPath(fnameOut), arrDRjoint)
    except Exception as e:
        print(e)

# ==============================================================================================================
# Entry point
# ==============================================================================================================

def findUnprocessedRecordings(dir, overwrite=False):
    pat = os.path.join(dir, '**/wavelet*.npy').replace('\\', '/')
    fnames = [x.replace('\\','/') for x in tqdm(glob.iglob(pat, recursive=True), desc='enumerating files')]
    pat = os.path.join(dir, '**/rawmvmt*.npy').replace('\\', '/')
    fnames+= [x.replace('\\','/') for x in tqdm(glob.iglob(pat, recursive=True), desc='enumerating files')]
    fnames = [x for x in fnames if ('tsne' in x or 'umap' in x)]
    fnames = [x for x in fnames if 'old' not in x and 'copy' not in x.lower()]
    fnames = [x for x in fnames if 'croprot' not in x and ('/wavelet/' in x or '/embeddings/' in x)]
    fnames = [x for x in fnames if not x.endswith('.filtered.npy') and not x.endswith('.filtered2.npy') and not x.endswith('.clusters.npy') and not x.endswith('.watershed.npy')]
    fnames = [x for x in fnames if 'wavelet_' not in os.path.abspath(os.path.join(x, '../'))]
    fnames = [x for x in fnames if 'rawmvmt_' not in os.path.abspath(os.path.join(x, '../'))]

    if not overwrite:
        fnames = [x for x in fnames if not os.path.exists(x.replace('.npy','') + '.filtered.npy')]

    return fnames

def filterFileSafe(fn, showFullProgress = True):
    try:
        filterFile(fn, showFullProgress = showFullProgress)
    except Exception as e:
        print(fn, e)

if __name__ == "__main__":

    from pipeline.python.misc import gui_misc

    r = gui_misc.askUser('File or directory?', 'Do you want to process a single file (Yes) or a whole directory (No)?')

    if r:
        selectedFile = gui_misc.askUserForFile()
        filterFile(selectedFile)
    else:
        selectedDir, OVERWRITE_RECORDING, nRec, fnames = gui_misc.askUserForDirectory(
            findUnprocessedRecordings, returnFilenames=True, forceOverwrite=False)

        NUM_PARALLEL = 58
        Parallel(n_jobs = NUM_PARALLEL)(delayed(filterFileSafe)(fn, showFullProgress=False) for fn in tqdm(fnames))