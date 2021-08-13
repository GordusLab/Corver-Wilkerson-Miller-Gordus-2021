# Reference for length of limbs
FNAME_LIMB_LENGTHS_REFERENCE = 'D:/data/bounds.pickle'

# =====================================================================================
# Imports & Globals
# =====================================================================================

# Add 'libraries' directory to path (to allow LEAP to be imported)
import os, sys
ROOT = os.getcwd()[:os.getcwd().rfind('spider-behavior') +  len('spider-behavior')];
sys.path.append(os.path.join(ROOT,'libraries/'))

# Import all libraries
import os, numpy as np, scipy.io, scipy.ndimage, pickle, gc, logging, matplotlib.pyplot as plt, joblib as jl, pandas as pd
from numba import jit
from tqdm import tqdm as tqdm
from joblib import Parallel, delayed
import statsmodels.regression.linear_model
from itertools import chain, combinations
from motmot.SpiderMovie import SpiderMovie

# Set process priority to lowest so this script doesn't interfere with OS function
import psutil
p = psutil.Process()
try:
    p.nice(psutil.IDLE_PRIORITY_CLASS)
except:
    p.nice(20)

# =====================================================================================
# Coordinate system transformation function
# =====================================================================================

@jit(nopython=True)
def transformLeap(arrLeap, arrMat):
    leapTrf = np.full(arrLeap.shape, np.nan, dtype=arrLeap.dtype)

    for r in range(arrLeap.shape[0]):
        for i in range(arrLeap.shape[2]):
            # Create rotation matrix
            theta = arrMat[r, 3] * np.pi / 180
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)), dtype=np.double)
            # Transform
            leapTrf[r, 0:2, i] = (arrLeap[r, 0:2, i] - np.array([100, 100], dtype=np.double)) @ R.T + \
                                 np.array([arrMat[r, 1], arrMat[r, 2]], dtype=np.double)
            # Save third column of LEAP array as well
            leapTrf[r, 2, i] = arrLeap[r, 2, i]

    return leapTrf

# =====================================================================================
# Limb-Length Filtering
# =====================================================================================

# We start by defining the skeleton (which joints are connected)
JOINT_SETS = [
    (2, 6), (6, 10),
    (14, 18), (18, 22),
    (3, 7), (7, 11),
    (15, 19), (19, 23),
    (4, 8), (8, 12),
    (16, 20), (20, 24),
    (5, 9), (9, 13),
    (17, 21), (21, 25)
]

# Determine max number of partner joints of any joint
def getMaxPartners():
    partners = {}
    for segm in JOINT_SETS:
        for a, b in [segm, (segm[1], segm[0])]:
            if a not in partners:
                partners[a] = [b]
            else:
                partners[a].append(b)

    return np.max([len(x) for x in partners.items()])

# Compute distances (N) and angles (N-1) of a given sequence of joints
def _computeDistanceAngles(poseData, jointIDs):
    dists = []
    angles = []
    # Compute distances
    for i in range(1, len(jointIDs)):
        dists.append(np.linalg.norm(poseData[jointIDs[i], :] - poseData[jointIDs[i - 1], :]))
    # Compute angles
    for i in range(2, len(jointIDs)):
        v1 = poseData[jointIDs[i], :] - poseData[jointIDs[i - 1], :]
        v2 = poseData[jointIDs[i - 1], :] - poseData[jointIDs[i - 2], :]
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        angles.append(np.arccos(np.dot(v1, v2)))

    return dists, angles

# Compute distances and angles for all frames in a matrix
def computeDistanceAngles(poseData, jointIDs):
    dists = []
    angles = []
    for i in range(poseData.shape[0]):
        x = _computeDistanceAngles(poseData[i, :, :], jointIDs)
        dists.append(x[0])
        angles.append(x[1])
    return np.array(dists), np.array(angles)


# Compute distances of partners, return as 3D matrix
# Input:
#     arr: matrix of size N (e.g. 10k) x #joints (e.g. 26) x Maximum number of partner joints
def computeSegmentLengths(arr):
    # Compute distances for all segments
    # -- Manually create memory-mapped temporary file
    fnameTmp = 'C:/DATA/segmentlength_{}.tmp'.format(np.random.randint(0, 999999999))
    while os.path.exists(fnameTmp):
        fnameTmp = 'C:/DATA/segmentlength_{}.tmp'.format(np.random.randint(0, 999999999))
    os.makedirs(os.path.dirname(fnameTmp), exist_ok=True)
    jl.dump(arr, fnameTmp)
    d = jl.load(fnameTmp, mmap_mode='r')

    # Process in parallel
    results = Parallel(n_jobs=30, prefer='processes')(delayed(computeDistanceAngles)(
        d, joints) for joints in JOINT_SETS)

    # Delete temporary file
    try:
        del d
        gc.collect()
        os.remove(fnameTmp)
    except Exception as e:
        print(e)

    # Extract distances
    dists = [x[0] for x in results]

    # Create matrix to hold segment lengths to all partners (populate with NaN to start)
    # Matrix: NumFrames x NumJoints x MaxNumPartnerJoints
    distsArr = np.full((arr.shape[0], arr.shape[1], getMaxPartners()), np.nan, dtype=np.float32)

    # Loop through segment lengths, filling the distance matrix
    for i in range(len(dists)):
        # The 2 indices of the labels that this segment consists of
        a, b = JOINT_SETS[i]
        # Use the first block of the 3D matrix that is not used yet
        pidx1 = np.argwhere(np.all(np.isnan(distsArr[:, a, :]), axis=0))[0][0]
        pidx2 = np.argwhere(np.all(np.isnan(distsArr[:, b, :]), axis=0))[0][0]
        # Add the distances to the newly formatted matrix (both ways)
        distsArr[:, a, pidx1] = dists[i][:, 0]
        distsArr[:, b, pidx2] = dists[i][:, 0]

    # Done! Return result...
    return distsArr

# Determine which joints are valid based on segment lengths
def validJoints(dists, useManualReference = False):
    # Get segment length boundsbased on human-annotated dataset
    bounds = None
    if useManualReference:
        bounds = None
        with open(FNAME_LIMB_LENGTHS_REFERENCE, 'rb') as f:
            bounds = pickle.load(f)
    else:
        # Compute bounds based on mean/std of limb lengths
        bounds = np.full((26, 2, 2), np.nan, dtype=np.float)

        for limbID in range(26):
            for k in range(2):
                ll = dists[:, limbID, k].copy()
                ll = ll[~np.isnan(ll)]
                if ll.size > 0:
                    H, edg = np.histogram(ll, bins=100)
                    edg = edg[1:] * 0.5 + edg[:edg.size - 1] * 0.5

                    m = edg[np.argmax(H)]
                    H_ = np.hstack((H[edg < m], m + np.flip(H[edg <= m])))
                    edg_ = np.hstack((edg[edg < m], np.flip(edg[edg <= m])))
                    a = np.hstack([np.repeat(e, h) for h, e in zip(H_, edg_)])

                    bounds[limbID, k, 0] = np.mean(a) - 5 * np.std(a)
                    bounds[limbID, k, 1] = np.mean(a) + 8 * np.std(a)

    # Initialize results matrix
    jointValid = np.full((dists.shape[0], dists.shape[1]), True, dtype=np.bool)

    for k1 in range(bounds.shape[0]):
        for k2 in range(bounds.shape[1]):
            # Get bounds
            bl, bu = bounds[k1, k2, 0], bounds[k1, k2, 1]
            if not np.any(np.isnan(bounds[k1, k2, :])):
                # Only label segments as valid in predicted dataset if they're within this bound
                notvalid = (dists[:, k1, k2] < bl) | (dists[:, k1, k2] > bu)
                jointValid[notvalid, k1] = False

    # Done!
    return jointValid

# =====================================================================================
# Impute datapoints from regression
# Note: This is useful when there are large gaps where joints are not detected
# =====================================================================================

def getTrainingAndPredictionSubsets(data, coordToPred, coordsIndep, allIndepVars):
    # Now determine what rows require this prediction
    # (i.e. that have all the independent variables but lack the dependent one)
    subsetPrediction = np.isnan(data[:, coordToPred[0], coordToPred[1]])
    for coordIndep in coordsIndep:
        subsetPrediction = np.logical_and(subsetPrediction, ~np.isnan(data[:, coordIndep[0], coordIndep[1]]))

    # Don't use this prediction for rows that have additional coordinates available, as they are
    # already covered by the regressions with more variables
    for coordIndepMissing in [x for x in allIndepVars if not x in coordsIndep]:
        subsetPrediction = np.logical_and(subsetPrediction, np.isnan(data[:, coordIndepMissing[0], coordIndepMissing[1]]))

    # Select training rows (that contain the independent and all independent variables (no NaNs))
    subsetTraining = ~np.isnan(data[:, coordToPred[0], coordToPred[1]])
    for coordIndep in coordsIndep:
        subsetTraining = np.logical_and(subsetTraining, ~np.isnan(data[:, coordIndep[0], coordIndep[1]]))

    return subsetPrediction, subsetTraining

def predictOLS(data, limbID, coordToPred, coordsIndep, coordsIndepAll):
    # Get prediction/training subset, to make sure this imputation regression is necessary/feasible
    subsetPrediction, subsetTraining = getTrainingAndPredictionSubsets(
        data, coordToPred, coordsIndep, coordsIndepAll)

    if np.sum(subsetTraining) > 0 and np.sum(subsetPrediction) > 0:

        # Build training matrix (of independent variables) (containing 1st and 2nd order terms + intercept)
        arrTraining = np.full((np.sum(subsetTraining), len(coordsIndep) * 2 + 1), 1)
        for i, coordIndep in enumerate(coordsIndep):
            arrTraining[:, 1 + i * 2 + 0] = data[subsetTraining, coordIndep[0], coordIndep[1]]
            arrTraining[:, 1 + i * 2 + 1] = data[subsetTraining, coordIndep[0], coordIndep[1]] ** 2

        # Build prediction matrix (of independent variables) (containing 1st and 2nd order terms + intercept))
        arrPred = np.full((np.sum(subsetPrediction), len(coordsIndep) * 2 + 1), 1)
        for i, coordIndep in enumerate(coordsIndep):
            arrPred[:, 1 + i * 2 + 0] = data[subsetPrediction, coordIndep[0], coordIndep[1]]
            arrPred[:, 1 + i * 2 + 1] = data[subsetPrediction, coordIndep[0], coordIndep[1]] ** 2

        # Run training regression
        ols = statsmodels.regression.linear_model.OLS(
            data[subsetTraining, coordToPred[0], coordToPred[1]], arrTraining).fit()

        # Use the model to predict missing values
        pred = ols.predict(arrPred)

        # Return results
        return (ols, pred, subsetPrediction, coordToPred[0], coordToPred[1])
    else:
        return None

# Source: https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset/54288550
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def imputeCoordinates(data):

    # Define all 8 limbs
    limbs = [
        (0, 14, 18, 22),
        (0, 15, 19, 23),
        (0, 16, 20, 24),
        (0, 17, 21, 25),
        (0, 5, 9, 13),
        (0, 4, 8, 12),
        (0, 3, 7, 11),
        (0, 2, 6, 10)
    ]

    configs = []

    for limbID in tqdm(range(len(limbs)), leave=False):

        for coordToPred in [(x, i) for i in range(2) for x in limbs[limbID]]:

            allIndepVars = [(x, i) for i in range(2) for x in limbs[limbID] if (x, i) != coordToPred]
            allIndepCoordCombinations = list(powerset(allIndepVars))

            for coordsIndep in allIndepCoordCombinations:
                configs.append((limbID, coordToPred, coordsIndep, allIndepVars))

    logging.info('Processing {} limb imputations.'.format(len(configs)))

    # Predict in parallel
    dataInterp = data.copy()
    # Save to temporary file
    fnameTmp = 'C:/DATA/imputecoords_{}.tmp'.format(np.random.randint(0, 999999999))
    while os.path.exists(fnameTmp):
        fnameTmp = 'C:/DATA/imputecoords_{}.tmp'.format(np.random.randint(0, 999999999))
    os.makedirs(os.path.dirname(fnameTmp), exist_ok=True)
    jl.dump(data, fnameTmp)
    d = jl.load(fnameTmp, mmap_mode='r')
    for r in Parallel(n_jobs=60)(
            delayed(predictOLS)(d, limbID, coordToPred, coordsIndep, allIndepVars) for
            limbID, coordToPred, coordsIndep, allIndepVars in tqdm(configs)):
        if r is not None:
            ols, pred, subsetPrediction, coordToPred1, coordToPred2 = r
            # Save result
            dataInterp[subsetPrediction, coordToPred1, coordToPred2] = pred
    # Delete temporary file
    try:
        del d
        gc.collect()
        os.remove(fnameTmp)
    except Exception as e:
        print(e)
    # Clear memory
    gc.collect()

    return dataInterp

# =====================================================================================
# Interpolate spider tracking so there are no NaNs left (required for wavelet)
# =====================================================================================

@jit(nopython=True, nogil=True)
def applyRotationAlongAxis(R, X):
    """
    This helper function applies a rotation matrix to every <X, Y> position tuple in a Nx2 matrix.
    Note: Numba JIT leads to a ~6-fold speed improvement.
    """
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j, 0:2] = R[:, :, i] @ X[i, j, 0:2]

def applyRotation(theta, X):
    # Create rotation matrix
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    # Perform rotation (Takes on the order of 15-30 seconds for most datasets.)
    applyRotationAlongAxis(R, X)

def runInterpolation(fname, algorithm, showWarnings = False):
    # Load data
    data = np.load(fname)

    # Determine output filename
    fnameOut = fname.replace('.npy','') + '_interp.npy'

    # Apply interpolation to front and back joints (they are used to compute position and orientation
    for i in range(len([0, 1])):
        for j in range(2):
            x0 = np.arange(data.shape[0])
            y0 = data[:, i, j]
            na = ~np.isnan(y0)

            try:
                f = scipy.interpolate.interp1d(x0[na], y0[na], kind='nearest', bounds_error=False,
                                               fill_value=np.array([np.nan, ]))
                y1 = f(x0)
            except:
                # If the column contains too many or all NaNs, it can be impossible to interpolate and interp1d will throw an error
                if showWarnings:
                    import warnings
                    warnings.warn('Error during interpolation of {} coordinate {}'.format(i, j))

            # First try replacing NaN's with right-most value
            # Ref: https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
            mask = np.isnan(y1)
            idx = np.where(~mask, np.arange(mask.shape[0]), 0)
            np.maximum.accumulate(idx, axis=0, out=idx)
            y1[mask] = y1[idx[mask]]

            # Now do the same in reverse direction
            y1 = np.flip(y1)
            mask = np.isnan(y1)
            idx = np.where(~mask, np.arange(mask.shape[0]), 0)
            np.maximum.accumulate(idx, axis=0, out=idx)
            y1[mask] = y1[idx[mask]]
            y1 = np.flip(y1)

            data[:, i, j] = y1

            if np.any(np.isnan(data[:, i, j])):
                import warnings
                warnings.warn('NaNs left for joint {} coordinate {}. This should not happen.'.format(i, j))

    # Center the data
    xy = np.mean(data[:, [0, 1], :], axis=1)
    datarel = data - np.repeat(xy[:, np.newaxis, :], 26, axis=1)

    # Find orientation of spider
    v = datarel[:, 0, :] - datarel[:, 1, :]
    theta = np.arctan2(v[:, 0], v[:, 1])

    # Orient spider in same direction for interpolation
    applyRotation(theta, datarel)

    # Apply interpolation to every joint
    for i in tqdm(range(datarel.shape[1]), leave=False):
        for j in range(datarel.shape[2]):
            x0 = np.arange(datarel.shape[0])
            y0 = datarel[:, i, j]
            na = ~np.isnan(y0)

            # Compute timepoints not more than 10 frames away from an existing point
            toInterpolate = scipy.ndimage.binary_dilation(np.isnan(y0), iterations=10)

            try:
                f = scipy.interpolate.interp1d(x0[na], y0[na], kind='linear', bounds_error=False,
                                               fill_value=np.array([np.nan, ]))
                y1 = f(x0)
            except:
                # If the column contains too many or all NaNs, it can be impossible to interpolate and interp1d will throw an error
                if showWarnings:
                    import warnings
                    #warnings.warn('Error during interpolation of {} coordinate {}'.format(i, j))

            datarel[toInterpolate, i, j] = y1[toInterpolate]

            if np.any(np.isnan(datarel[:, i, j])):
                import warnings
                #warnings.warn('NaNs left for joint {} coordinate {}. This should not happen.'.format(i, j))

    # Now apply regression to infer points that are far away from another measurement
    datarelImp = imputeCoordinates(datarel)
    datarel = datarelImp

    # Rotate spider back to its original orientation
    applyRotation(-theta, datarel)

    # Translate spider back to its original position
    data = datarel + np.repeat(xy[:, np.newaxis, :], 26, axis=1)

    # Save
    np.save         (fnameOut, data)
    scipy.io.savemat(fnameOut.replace('.npy', '') + '.mat', {'{}_abs_filt_interp'.format(algorithm): data})

    # Return data
    return data

# =====================================================================================
# Filter out frames where the spider is not moving (useful for wavelet transform)
# =====================================================================================

def runFilterStationaryFrames(fname, algorithm):
    # Load filteted, interpolated, absolute data
    fnameData = fname.replace('.npy','') + '_abs_filt_interp.npy'
    data = np.load(fnameData)

    # Compute limb velocities
    vel = np.abs(data - np.roll(data, 10,  axis=0))
    # Filter out velocities larger than 100 (they're errors)
    vel[vel > 100] = 0

    # Filter out low-velocities
    filt = scipy.ndimage.binary_dilation(np.any(np.any(vel[:, :, 0:2] >= 25, axis=2), axis=1), iterations=100)
    print('Kept {} fraction of data after filtering static frames'.format(np.mean(filt)))

    # Also filter out border
    filtBorder = np.logical_and(
        np.all(np.all(data[:, :, 0:2] > 100, axis=1), axis=1),
        np.all(np.all(data[:, :, 0:2] < 924, axis=1), axis=1))

    print('Kept {} fraction of data after filtering static frames and border'.format(
        np.mean(np.logical_and(filt, filtBorder))))

    filtNoBorder = np.logical_and(filt, filtBorder)

    #
    dataMvmt         = data[filt,:,:]
    dataMvmtNoBorder = data[filtNoBorder,:,:]

    # Save
    fnameOut = fnameData.replace('.npy','') + '_mvmt.npy'
    np.save(fnameOut, dataMvmt)
    np.save(fnameOut.replace('.npy','') + '.idx.npy', filt)

    fnameOut = fnameData.replace('.npy','') + '_mvmt_noborder.npy'
    np.save(fnameOut, dataMvmtNoBorder)
    np.save(fnameOut.replace('.npy','') + '.idx.npy', filtNoBorder)

# =====================================================================================
# Main function
# =====================================================================================

def processRecording(fname, overwrite=False):
    for algorithm in ['dlc', 'leap']:
        if overwrite or not os.path.exists(getOutputFilename(fname, algorithm)):
            processRecordingAlgorithm(fname, algorithm=algorithm, overwrite=overwrite)

def processRecordingAlgorithm(fname, algorithm='leap', overwrite=False, returnLimbLengthsOnly = False):

    if algorithm not in ['dlc', 'leap']:
        raise Exception('Unknown algorithm specified: {}'.format(algorithm))

    # TEMPORARY:
    if algorithm != 'dlc':
        return

    if fname is None:
        return # Empty filename, so return

    # Get output filenames
    fnameLeapFilt = getOutputFilename(fname, algorithm=algorithm)

    # Already exists?
    if not overwrite and os.path.exists(fnameLeapFilt):
        return # Already exists, exit

    # Get filenames
    dirLeap = os.path.dirname(fname)
    try:
        fnameLeap = [os.path.join(dirLeap, x) for x in os.listdir(dirLeap) if x.endswith('_{}.npy'.format(algorithm))][0]
        fnameMat = [os.path.join(dirLeap, x) for x in os.listdir(dirLeap) if x.endswith('_mat.npy')][0]
    except:
        return # Files do not exist

    fnameAbsFiltNpy = fnameLeap.replace('_{}.npy'.format(algorithm), '') + '_{}_abs_filt.npy'.format(algorithm)
    fnameAbsFiltMat = fnameLeap.replace('_{}.npy'.format(algorithm), '') + '_{}_abs_filt.mat'.format(algorithm)

    print('[{}] Starting'.format(fname[-200:]))

    # Determine arrays
    a = np.memmap(fnameLeap, mode='r', dtype=np.float32)
    N = int(a.shape[0] / (26 * 3))
    del a

    # Open arrays
    arrLeap = np.memmap(fnameLeap, mode='r', shape=(N, 3, 26), dtype=np.float32).copy()
    arrMat  = np.memmap(fnameMat , mode='r', shape=(N, 4)    , dtype=np.double ).copy()

    # Get transformed LEAP coordinates
    print('[{}] Transforming coordinates'.format(fname[-200:]))
    leapTrf = transformLeap(arrLeap, arrMat).transpose((0, 2, 1))

    print('[{}] Saving coordinates'.format(fname[-200:]))

    # Save transformed (unfiltered) coordinates
    np.save         (fnameLeap.replace('_{}.npy'.format(algorithm), '') + '_{}_abs.npy'.format(algorithm), leapTrf)
    scipy.io.savemat(fnameLeap.replace('_{}.npy'.format(algorithm), '') + '_{}_abs.mat'.format(algorithm), {'{}_abs'.format(algorithm): leapTrf})

    # Save original coordinates in Matlab format also
    scipy.io.savemat(fnameLeap.replace('.npy', '') + '.mat', {'{}'.format(algorithm): arrLeap})
    scipy.io.savemat(fnameMat .replace('.npy', '') + '.mat', {'posrot': arrMat })

    # Compute limb lengths
    print('[{}] Computing Limb Lengths'.format(fname[-200:]))
    limbLengths = computeSegmentLengths(leapTrf)

    # Debug option: return limb lengths and end
    if returnLimbLengthsOnly:
        return limbLengths

    # Determine which joints are valid based on limb lengths
    jointValid = validJoints(limbLengths)

    # Remove joints that violate limb lengths
    leapFilt = leapTrf.copy()
    leapFilt[~jointValid, :] = np.nan

    # Save the transformed, filtered coordinates
    print('[{}] Saving filtered coordinates'.format(fname[-200:]))
    np.save         (fnameAbsFiltNpy, leapFilt)
    scipy.io.savemat(fnameAbsFiltMat, {'{}_abs_filt'.format(algorithm): leapFilt})

    # Now compute the position & orientation
    JOINTS_FRONT = [0, ]  # [2, 0, 14]
    JOINTS_BACK = [1, ]  # [5, 1, 17]

    jointFront = np.mean(leapFilt[:, JOINTS_FRONT, 0:2], axis=1)
    jointBack = np.mean(leapFilt[:, JOINTS_BACK, 0:2], axis=1)

    # -- Compute the thorax center position
    pos = np.mean(leapFilt[:, JOINTS_FRONT + JOINTS_BACK, 0:2], axis=1)
    # -- Compute the thorax orientation
    v = jointFront - jointBack
    orientation = np.arctan2(v[:, 0], v[:, 1])

    # Save the position/orientation to file in NumPy + Matlab formats
    posrot = np.hstack((pos, orientation[:,np.newaxis]))

    print('[{}] Saving position and orientation'.format(fname[-200:]))
    np.save         (fnameLeap.replace('_{}.npy'.format(algorithm), '') + '_{}_position_orientation.npy'.format(algorithm), posrot)
    scipy.io.savemat(fnameLeap.replace('_{}.npy'.format(algorithm), '') + '_{}_position_orientation.mat'.format(algorithm), {'{}_posrot'.format(algorithm): posrot})

    # Run interpolation on both DLC and Leap
    runInterpolation(fnameAbsFiltNpy, algorithm = algorithm)

    # Filter stationary frames
    runFilterStationaryFrames(fnameLeap, algorithm)
    
    # Now render a sample image
    print('[{}] Generating position plot'.format(fname[-200:]))
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(pos[::1, 0], pos[::1, 1], s=0.5, alpha=0.15)
    plt.xlim(0, 1024)
    plt.ylim(0, 1024)
    plt.savefig(fnameLeap.replace('_{}.npy'.format(algorithm), '') + '_{}_positions.png'.format(algorithm), dpi=250)
    plt.close(fig)
    print('[{}] Done'.format(fname[-200:]))

# =====================================================================================
# Helper functions
# =====================================================================================

def getRecordingFilename(d):
    fdir = os.path.join(d, 'croprot/')

    if not os.path.exists(fdir):
        return None

    fname = [os.path.join(fdir, x.name) for x in os.scandir(fdir) if x.name.endswith('_img.npy')]
    fname = None if len(fname) != 1 else fname[0]
    return fname

def getOutputFilename(fname, algorithm):
    return fname.replace('_img.npy', '') + '_{}_abs_filt.npy'.format(algorithm)

def findUnprocessedRecordings(rootpath, overwrite=True):
    dirs = []

    def _findUnprocessedRecordings(d):
        fx = os.path.join(rootpath, d)
        # Only proceed if this is a recording directory
        if isRecordingDir(fx):
            # Obtain cropped/rotated filename
            fname = getRecordingFilename(fx)
            if fname is not None:
                if overwrite or \
                        not os.path.exists(getOutputFilename(fname, 'leap')) or \
                        not os.path.exists(getOutputFilename(fname, 'dlc')):
                    return fname
        return None

    # Consider the directory itself also, in case it is a recording directory
    for x in Parallel(n_jobs=1)(delayed(_findUnprocessedRecordings)(d) for d in os.listdir(rootpath) + ['',]):
        if x is not None:
            dirs.append(x)

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
            # Process in random order
            np.random.shuffle(fnames)
            njobs = 1
            if njobs == 1:
                [processRecording(f, overwrite=OVERWRITE_RECORDING) for f in tqdm(fnames)]
            else:
                Parallel(n_jobs=njobs)(delayed(processRecording)(
                    f, overwrite=OVERWRITE_RECORDING) for f in tqdm(fnames))
