import os, numpy as np, regex as re, glob, psutil, logging, joblib as jl, scipy.optimize
from tqdm import tqdm
import numba as nb

def getWaveletFilename(settings):
    if 'fname_base' in settings:
        fdir = settings['fname_base']
    else:
        fdir = 'wavelet/'

    if not os.path.exists(fdir):
        os.makedirs(fdir, exist_ok=True)

    # List included joints
    if not 'joints' in settings:
        settings['joints'] = ''.join(['1' for _ in range(26)])

    # Determine key names to use for descriptor
    keys = ['joints','algorithm','coords','include_velocities','include_absolute_position',
            'rawmvmt_duration','rawmvmt_maxoffset','rawmvmt_meansubtract','rawmvmt_scalestd']
    nameKeys = [x for x in settings.keys() if x in keys]

    # Generate settings descriptor
    settingsDescr = '_'.join([ '{}'.format(settings[key]) for key in sorted(nameKeys)])

    fname = os.path.join(fdir, 'rawmvmt_{}.npy'.format(settingsDescr))

    return fname

def findTrackingFile(d, settings):
    baseName = os.path.join(d, 'croprot/*_{}.npy'.format(settings['algorithm']))
    #baseName = os.path.join(d, '*/*_{}_abs_filt_interp_mvmt_noborder.npy'.format(settings['algorithm']))
    g = glob.glob(baseName)
    if len(g) != 1:
        raise Exception('Could not find tracking file for {}'.format(d))
    else:
        return g[0]

def loadTrackingData(fnameTracking):
    try:
        A = np.load(fnameTracking)
        return A
    except:
        fnameTracking_ = '\\\\?\\' + fnameTracking.replace('\\\\?\\', '').replace('/', '\\')
        A = np.memmap(fnameTracking_, mode='r', dtype=np.float32)
        N = int(A.size // (26 * 3))
        assert((A.size % (26*3)) == 0)
        del A
        A = np.memmap(fnameTracking_, mode='r', dtype=np.float32, shape=(N, 3, 26))
        A_ = np.zeros(A.shape, dtype=np.float32)
        A_[:,:,:] = A[:,:,:]
        #A_ = np.copy(A)
        del A
        return A_

def computeEmbeddingData(settings):
    # Get overview of recordings
    from pipeline.python.behavioral_motifs.run_wavelet import listAvailableRecordings

    if 'recordings' in settings:
        fnames = settings['recordings']
    else:
        fnames = listAvailableRecordings()

    # Find tracking files
    fnamesTracking = [findTrackingFile(x, settings) for x in fnames]

    # Create shortened filename
    fnamesShort = [re.search('/behavior/([^/]*)/?', x.replace('\\', '/')).group(1) for x in fnames]

    # Determine how many rows to load
    nPerFile = [int(float(settings['maxN']) / len(fnames)) for i in range(len(fnames))]
    while np.sum(nPerFile) != settings['maxN']:
        nPerFile[np.argmin(nPerFile)] += 1

    # Joint IDs
    jointIDs = np.argwhere(np.array([1 if x == '1' else 0 for x in list(settings['joints'])])).T[0]

    # Currently either allow no-vel and joints, or no joints and with-vel
    if settings['include_velocities'] != 'no-vel' and len(jointIDs) > 0:
        raise Exception()

    # Determine number of columns
    ncol = len(jointIDs) * 2 * settings['rawmvmt_duration'] # Don't subsample
    if settings['include_velocities'] != 'no-vel':
        ncol += 3 * settings['rawmvmt_duration']

    # Allocate the output array
    out = np.zeros((np.sum(nPerFile), ncol), dtype=np.float32)

    from pipeline.python.behavioral_motifs.run_wavelet_cluster_newdata import getNewDataWaveletFilename
    FNAME_NEWDATA_WAV_BASE = getNewDataWaveletFilename(settings)

    # For each recording, select a random subset of rows
    c = 0
    for d, n, fnameTracking in tqdm(zip(fnames, nPerFile, fnamesTracking), leave=False):
        FNAME_NEWDATA_WAV = os.path.join(d, 'wavelet', FNAME_NEWDATA_WAV_BASE)
        if os.path.exists(FNAME_NEWDATA_WAV) and ('overwrite' not in settings or not settings['overwrite']):
            tmp = np.load(FNAME_NEWDATA_WAV)
        else:
            tmp = computeAllEmbeddingData(FNAME_NEWDATA_WAV, settings)
        out[c:(c+n),:] = tmp[np.random.choice(np.argwhere(np.all(np.logical_not(np.isnan(tmp)), axis=1)).T[0], n, replace=False), :]
        c += n

    # Save
    _fname = '\\\\?\\' + settings['fname'].replace('\\\\?\\', '').replace('/', '\\')
    os.makedirs(os.path.dirname(_fname), exist_ok=True)
    np.save(_fname, out)

@nb.jit(nopython=True, nogil=True)
def applyRotationAlongAxis(R, X):
    """
    This helper function applies a rotation matrix to every <X, Y> position tuple in a Nx2 matrix.
    Note: Numba JIT leads to a ~6-fold speed improvement.
    """
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, 0:2, j] = R[:, :, i] @ X[i, 0:2, j]

@nb.jit(nopython=True, nogil=True)
def applyRotationAlongAxis1d(R, X):
    """
    This helper function applies a rotation matrix to every <X, Y> position tuple in a Nx2 matrix.
    Note: Numba JIT leads to a ~6-fold speed improvement.
    """
    for i in range(X.shape[0]):
        X[i, 0:2] = R[:, :, i] @ X[i, 0:2]

def applyRotation(theta, X):
    # Create rotation matrix
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    # Perform rotation (Takes on the order of 15-30 seconds for most datasets.)
    if X.ndim == 3:
        applyRotationAlongAxis(R, X)
    elif X.ndim == 2:
        applyRotationAlongAxis1d(R, X)


def cost(ys, midPoints):  # y1, y2
    ptsX = midPoints[:, 0]
    ptsY = ys[0] + (ys[1] - ys[0]) * ptsX / 200.0
    dists = np.abs(midPoints[:, 1] - ptsY)
    if np.any((ptsY < 0) | (ptsY > 200)):
        return np.inf
    return np.max(dists)

def align(arr, jointIDs, returnOpt=False):
    jointPairs = [
        (13, 25), (9, 21), (5, 17), (4, 16), (8, 20), (12, 24), (3, 15), (7, 19), (11, 23), (6, 18), (10, 22)
    ]

    midPoints = np.array([0.5 * (arr[jp[0], :] + arr[jp[1], :]) for
                          jp in jointPairs if jp[0] in jointIDs or jp[1] in jointIDs])

    opt = scipy.optimize.minimize(lambda x: cost(x, midPoints),
                                  np.ones(2) * 100, method='Nelder-Mead', options={'maxiter': 1000})
    y1, y2 = opt.x

    v1 = np.array([200, y2 - y1])
    v1 /= np.linalg.norm(v1)

    v2 = np.array([v1[1], -v1[0]])

    arr2 = np.zeros(arr.shape, dtype=arr.dtype)
    arr2[:, 0] = arr @ v1
    arr2[:, 1] = arr @ v2

    center = np.mean(arr2[[0, 1, 2, 3, 4, 5, 14, 15, 16, 17], 0:2], axis=0) - 100

    arr2 -= center[np.newaxis, :]

    if returnOpt:
        return arr2, opt
    else:
        return arr2

def alignBlock(arr, jointIDs):
    return [align(x, jointIDs) for x in arr]

def alignTrackingByMidline(arrTracking, jointIDs):
    arrTrackingList = list(chunks([arrTracking[i, 0:2, :].T for i in range(arrTracking.shape[0])], 5000))
    arrTrackingListFix = jl.Parallel(n_jobs=55)(jl.delayed(alignBlock)(ch, jointIDs) for ch in tqdm(arrTrackingList))
    arrTrackingFix2 = np.vstack(arrTrackingListFix)
    arrTrackingFix = np.zeros(arrTracking.shape, dtype=arrTracking.dtype)
    for i in range(arrTracking.shape[0]):
        arrTrackingFix[i, 0:2, :] = arrTrackingFix2[i, :, :].T

    return arrTrackingFix

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def computeAllEmbeddingData(fnameOut, settings):
    if os.path.exists(fnameOut) and not settings['overwrite']:
        return

    fname = os.path.dirname(os.path.dirname(fnameOut))
    #if 'recordings' in settings:
    #    fname = settings['recordings'][0]

    # Find tracking files
    fnameTracking = findTrackingFile(fname, settings)
    jointIDs = np.argwhere(np.array([1 if x == '1' else 0 for x in list(settings['joints'])])).T[0]

    # Currently either allow no-vel and joints, or no joints and with-vel
    if settings['include_velocities'] != 'no-vel' and len(jointIDs) > 0:
        raise Exception()

    # Determine number of columns
    ncol = len(jointIDs) * 2 * settings['rawmvmt_duration'] # Don't subsample
    if settings['include_velocities'] != 'no-vel':
        ncol += 3 * settings['rawmvmt_duration']

    # Determine number of columns
    arrTracking = loadTrackingData(fnameTracking)

    # Fix tracking data alignment?
    if settings['coords'] == "euclidean-midline":
        arrTracking = alignTrackingByMidline(arrTracking, jointIDs)

    # Allocate the output array
    out = np.zeros((arrTracking.shape[0], ncol), dtype=np.float32)

    # event radius
    rad = settings['rawmvmt_duration'] // 2

    # Polar coordinates?
    JOINT_PARTNERS = {
        13: 9, 9: 5,
        12: 8, 8: 4,
        11: 7, 7: 3,
        10: 6, 6: 2,
        22: 18, 18: 14,
        23: 19, 19: 15,
        24: 20, 20: 16,
        25: 21, 21: 17,
        5: 0, 4: 0, 3:0, 2:0, 14:0, 15: 0, 16: 0, 17: 0, 1:0, 0:1
    }

    # Remove baseline?
    removeBaseline = True
    removeBaselineMode = 'event' # [event, all]
    scaleStd = False

    if 'rawmvmt_meansubtract' in settings:
        removeBaseline = (settings['rawmvmt_meansubtract'] != 'nomeansub')
        if settings['rawmvmt_meansubtract'] == 'meansuball':
            removeBaselineMode = 'all'

    scaleTest = False
    if 'rawmvmt_scalestd' in settings:
        scaleStd = (settings['rawmvmt_scalestd'] in ['scalestd', 'scaletest'])
        assert(scaleStd in [True, False])
        if settings['rawmvmt_scalestd'] == 'scaletest':
            scaleTest = True

    # Check order of dimensions
    assert(arrTracking.shape[1] in [2,3])

    logging.warn('Started processing {}'.format(fnameOut))

    if settings['include_velocities'] == 'with-vel-polar':
        # Compute XY velocity
        xy = np.mean(arrTracking[:, 0:2, [0, 1]], axis=2)
        xyVel = xy - np.roll(xy, 10, axis=0)
        xyVel[0:10,:] = np.nan
        # Compute angular velocity
        v = arrTracking[:, 0:2, 0] - arrTracking[:, 0:2, 1]
        v /= np.linalg.norm(v, axis=1)[:, np.newaxis]
        theta =  np.arctan2(v[:, 0], v[:, 1])
        from pipeline.python.behavioral_motifs.run_wavelet import thetaToHalfCircle
        thetaVel = np.vectorize(thetaToHalfCircle)(theta - np.roll(theta, 10))
        thetaVel[0:10] = np.nan
        # Compute forward/sideway velocity
        xyFS = np.full(xyVel.shape, np.nan, dtype=np.float32)
        vOrth = np.array([v[:,1], -v[:,0]], dtype=v.dtype).T
        xyFS[:,0] = np.einsum('ij, ij -> i', xyVel, v)
        xyFS[:,1] = np.einsum('ij, ij -> i', xyVel, vOrth)
        # Append
        dAllLimbs = np.full((xyFS.shape[0], 3), np.nan, dtype=np.float32)
        dAllLimbs[:,0:2] = xyFS
        dAllLimbs[:,2] = thetaVel
        # ...
        stdLimbs = np.nanstd(dAllLimbs, axis=0)[np.newaxis, :]
        for idx in tqdm(range(rad, arrTracking.shape[0] - (rad)), leave=False, mininterval=5):
            d = dAllLimbs[(idx - rad + 0):(idx + rad), :]
            dMean = np.mean(d, axis=0)
            d = d - dMean
            if scaleStd:
                d = d / stdLimbs
            if not removeBaseline:
                d = d + dMean
            out[idx, :] = d.reshape((-1,))

    elif settings['include_velocities'] == 'no-vel':
        if settings['coords'] == 'polar':
            dAllLimbs = arrTracking[:, 0:2, :].copy()
            for j in JOINT_PARTNERS.keys():
                dAllLimbs[:, :, j] -= arrTracking[:, 0:2, JOINT_PARTNERS[j]]
                # Rotate such that the joint partner is fixed in orientation
                p2 = arrTracking[:, 0:2, JOINT_PARTNERS[JOINT_PARTNERS[j]]]
                p1 = arrTracking[:, 0:2, JOINT_PARTNERS[j]]
                p0 = arrTracking[:, 0:2, j]

                v1 = (p1-p2) / np.linalg.norm(p1-p2, axis=1)[:,np.newaxis]
                v2 = (p0-p1) / np.linalg.norm(p0-p1, axis=1)[:,np.newaxis]
                thetas = np.arctan2(v1[:,0], v1[:,1]) - np.arctan2(v2[:,0], v2[:,1])
                thetas[thetas < -np.pi] += np.pi * 2
                thetas[thetas >  np.pi] -= np.pi * 2
                lengths = np.linalg.norm(p0-p1, axis=1)

                # Switch to polar coordinate system if settings indicate it
                dAllLimbs[:, 0, j] = thetas
                dAllLimbs[:, 1, j] = lengths

            stdLimbs = np.std(dAllLimbs, axis=0)[np.newaxis, 0:2, jointIDs]
            for idx in tqdm(range(rad, arrTracking.shape[0]-(rad)), leave=False, mininterval=5):
                d = dAllLimbs[(idx - rad + 0):(idx + rad), 0:2, jointIDs]
                dMean = np.mean(d, axis=0)
                d = d - dMean
                if scaleStd:
                    d = d / stdLimbs
                if not removeBaseline:
                    d = d + dMean
                out[idx, :] = d.reshape((-1,))
        else:
            # For each recording, select a random subset of rows
            stdLimbs = np.std(arrTracking, axis=0)[np.newaxis, 0:2, jointIDs]
            dMeanAll = np.mean(arrTracking, axis=0)[0:2, jointIDs]
            for idx in tqdm(range(rad, arrTracking.shape[0]-(rad)), leave=False, mininterval=5):
                # Remove baseline
                d = arrTracking[(idx - rad + 0):(idx + rad), 0:2, jointIDs]
                dMean = np.mean(d, axis=0)
                d = (d - dMeanAll) if removeBaselineMode == 'all' else (d - dMean)
                if scaleStd:
                    d = d / stdLimbs
                if not removeBaseline:
                    d = d + dMean
                # Temporary test! Weigh back legs much more heavily to ensure the cluster is split
                if scaleTest:
                    d[:, :, 4:] *= 50
                out[idx, :] = d.reshape((-1,))

    # Only use moving timepoints
    fnameIdx = [x for x in glob.glob(os.path.join(os.path.dirname(fnameTracking), '*_noborder.idx.npy')) if 'dlc' in x][0]
    arrIdx = np.load(fnameIdx)
    out = out[arrIdx,:]

    out[0:rad, :] = np.nan
    out[arrIdx.shape[0] - rad:, :] = np.nan

    # Save
    logging.warn('Started saving array ({}) to {}'.format(out.shape, fnameOut))
    os.makedirs(os.path.dirname(fnameOut), exist_ok=True)
    np.save(fnameOut, out)
    logging.warn('Saved array ({}) to {}'.format(out.shape, fnameOut))

    # Return data
    return out

def getEmbeddingData(settings):
    if not os.path.exists(settings['fname']) or ('overwrite' in settings and settings['overwrite']):
        computeEmbeddingData(settings)

    A = np.load('\\\\?\\' + settings['fname'].replace('\\\\?\\', '').replace('/', '\\'))
    return A

def getMetricFunction(settings):

    nJoints = 0
    if isinstance(settings['joints'], str):
        nJoints = len(np.argwhere(np.array([1 if x=='1' else 0 for x in list(settings['joints'])])).T[0])
    else:
        nJoints = len(settings['joints'])

    maxOffset = 16 # 16 frames offset, i.e. 320 ms
    if 'rawmvmt_maxoffset' in settings:
        maxOffset = settings['rawmvmt_maxoffset']

    maxOffsetRad = maxOffset // 2
    s = settings['rawmvmt_duration'] - maxOffset
    nJointsSkip = nJoints * 2
    nVelSkip = 3

    @nb.njit(nogil=True, fastmath=True)
    def euclideanAdjustable_v2_vel(x_, y_):
        tc = 0
        for velID in range(3):
            bc = 1e9
            bc2 = 1e9
            for offset in range(maxOffset):
                c = 0
                c2 = 0
                i0 = offset * nVelSkip + velID
                i2 = maxOffsetRad * nVelSkip + velID
                for k in range(s):
                    i0 += nVelSkip
                    i2 += nVelSkip
                    c  += (x_[i0] - y_[i2]) ** 2
                    c2 += (y_[i0] - x_[i2]) ** 2
                if c < bc:
                    bc = c
                if c2 < bc2:
                    bc2 = c2
            tc += np.sqrt(bc) + np.sqrt(bc2)
        return 0.5 * tc

    @nb.njit(nogil=True, fastmath=True)
    def euclideanAdjustable_v2(x_, y_):
        tc = 0
        for jointID in range(nJoints):
            bc = 1e9
            bc2 = 1e9
            for offset in range(maxOffset):
                c = 0
                c2 = 0
                i0 = offset * nJointsSkip + jointID
                i1 = i0 + nJoints
                i2 = maxOffsetRad * nJointsSkip + jointID
                i3 = i2 + nJoints
                for k in range(s):
                    i0 += nJointsSkip
                    i1 += nJointsSkip
                    i2 += nJointsSkip
                    i3 += nJointsSkip
                    c  += (x_[i0] - y_[i2]) ** 2 + (x_[i1] - y_[i3]) ** 2
                    c2 += (y_[i0] - x_[i2]) ** 2 + (y_[i1] - x_[i3]) ** 2
                if c < bc:
                    bc = c
                if c2 < bc2:
                    bc2 = c2
            tc += np.sqrt(bc) + np.sqrt(bc2)
        return 0.5 * tc

    if settings['include_velocities'] != 'no-vel':
        return euclideanAdjustable_v2_vel
    else:
        return euclideanAdjustable_v2

# Old version
def getMetricFunction_v01(settings):

    nJoints = len(np.argwhere(np.array([1 if x=='1' else 0 for x in list(settings['joints'])])).T[0])
    maxOffset = 10 # 20 frames offset, i.e. 400 ms
    maxOffsetRad = maxOffset // 2
    s = settings['rawmvmt_duration'] // 2 - maxOffset

    @nb.njit(nogil=True, fastmath=True)
    def euclideanAdjustable(x_, y_):
        x = x_.reshape((-1, 2, nJoints))
        y = y_.reshape((-1, 2, nJoints))
        tc = 0
        for jointID in range(nJoints):
            bc = 1e9
            for offset in range(maxOffset):
                d = x[offset:(offset+s), :, jointID] - y[maxOffsetRad:(maxOffsetRad+s), :, jointID]
                c = np.sum(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])
                if c < bc:
                    bc = c
            tc += np.sqrt(bc)
        return tc

    return euclideanAdjustable
    #return 'euclidean'

if __name__ == "__main__":
    # Settings
    j1 = [6, 10, 18, 22, 9, 13, 21, 25]
    j2 = [10, 22, 13, 25]
    j3 = []

    settings = {
        'rawmvmt': True,
        'rawmvmt_duration': 100,
        'rawmvmt_maxoffset': 16,
        'rawmvmt_meansubtract': 'meansuball',
        'rawmvmt_scalestd': False,
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': j1,
        # Overwrite?
        'overwrite': False,
        # Coords
        'coords': 'euclidean',
        'use_pca': 'no-pca',
        'metric': 'euclidean',
        'include_absolute_position': 'no-abspos',
        'include_velocities': 'no-vel',
        'tsne_perplexities': [100, ],  # 100, 500
        'perplexity': 100,
        'perplexity_newdata': 100,
        'n_iter': 2000,
        'n_iter_newdata': 500,
        'maxN': 200000,
        'method': 'tsne'
    }

    # Report PID for debugging purposes and to allow user to change CPU resource allocation manually
    p = psutil.Process()
    pid = p.pid
    logging.info('Preparing rawmvmt embedding data (PID={})...'.format(pid))

    # Get available recordings
    from pipeline.python.behavioral_motifs.run_wavelet import listAvailableRecordings
    fnames = listAvailableRecordings()

    # TEMPORARY:
    #fnames = [x for x in fnames if '6-3-19-e' in x]
    print('fnames: {}'.format(fnames))

    # Get filenames
    from pipeline.python.behavioral_motifs.run_wavelet_cluster_newdata import getNewDataWaveletFilename

    #
    for j1 in [[6, 10, 18, 22], [9, 13, 21, 25]]: # [[0, 1, 6, 10, 18, 22], [0, 1, 9, 13, 21, 25]]: #
        refC = {
            'duration': 60, # 100
            'maxoffset': 16,
            'meansubtract': 'meansub',
            'scalestd': 'scalestd',
            'coords': 'euclidean-midline'
        }

        params = []
        for duration in [60, 100]:
            for maxoffset in [0, 16]:
                for meansubtract in ['meansub', 'nomeansub', 'meansuball']:
                    for scalestd in ['scalestd', 'noscalestd']:
                        for coords in ['polar', 'euclidean', 'euclidean-midline']:
                            # Only one property can deviate from the reference configuration, to avoid exponential increase
                            # of configurations
                            numDiff = 0
                            if duration != refC['duration']: numDiff += 1
                            if maxoffset != refC['maxoffset']: numDiff += 1
                            if scalestd != refC['scalestd']: numDiff += 1
                            if coords != refC['coords']: numDiff += 1
                            if meansubtract != refC['meansubtract']: numDiff += 1
                            if numDiff > 0:
                                continue

                            for d in tqdm(fnames):
                                settingsC = settings.copy()
                                settingsC['joints'] = j1
                                settingsC['coords'] = coords
                                settingsC['rawmvmt_duration'] = duration
                                settingsC['rawmvmt_maxoffset'] = maxoffset
                                settingsC['rawmvmt_meansubtract'] = meansubtract
                                settingsC['rawmvmt_scalestd'] = scalestd
                                FNAME_NEWDATA_WAV_BASE = getNewDataWaveletFilename(settingsC)
                                FNAME_NEWDATA_WAV = os.path.join(d, 'wavelet', FNAME_NEWDATA_WAV_BASE)
                                params.append((FNAME_NEWDATA_WAV, settingsC))

    logging.warn('Number of configs: {}'.format(len(params)))
    #jl.Parallel(n_jobs=min(1, len(params)+1))(jl.delayed(computeAllEmbeddingData)(*c) for c in params)
    jl.Parallel(n_jobs=1)(jl.delayed(computeAllEmbeddingData)(*c) for c in params)
