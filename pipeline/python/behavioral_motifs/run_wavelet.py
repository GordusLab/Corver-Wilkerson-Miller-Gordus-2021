
"""
    The following script applies wavelet transforms to behavioral recordings, and provides several convenience functions.
"""

# =====================================================================================
# Imports & Globals
# =====================================================================================

import gc, os, json, numpy as np, logging, scipy.io, joblib as jl, pandas as pd
from numba import jit
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import joblib as jl
from sklearn.decomposition import PCA

# Skeleton information
JOINTS_THORAX = [0, 1, 2, 3, 4, 5, 14, 15, 16, 17]

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

JOINTS_LEGS = [18, 22, 19, 23, 20, 24, 21, 25, 9, 13, 8, 12, 7, 11, 6, 10]

logging.getLogger().setLevel(logging.INFO)

# Set process priority to lowest so this script doesn't interfere with OS function
import psutil
p = psutil.Process()
try:
    p.nice(psutil.IDLE_PRIORITY_CLASS)
except:
    p.nice(20)

# =====================================================================================
# Helper functions
# =====================================================================================

def getWaveletFilename(d, settings):
    fdir = os.path.join(d, 'wavelet/')

    if not os.path.exists(fdir):
        os.makedirs(fdir, exist_ok=True)

    # List included joints
    if not 'joints' in settings:
        settings['joints'] = ''.join(['1' for _ in range(26)])

    # Small config check
    if 'legvars' in settings:
        settings['legvars_minF'] = settings['legvars']['minF']

    # Determine key names to use for descriptor
    keys = ['joints','numPeriods','algorithm','coords','include_velocities','include_absolute_position']
    if settings['legvars']['minF'] != 0.1:
        keys.append('legvars_minF')

    if settings['algorithm'] == 'rawpca':
        # Exclude irrelevant keys from filename
        keys = [x for x in keys if x not in ['joints','coords']]

    nameKeys = [x for x in settings.keys() if x in keys]

    # Generate settings descriptor
    settingsDescr = '_'.join([ '{}'.format(settings[key]) for key in sorted(nameKeys)])

    fname = os.path.join(fdir, 'wavelet_{}.npy'.format(settingsDescr))

    return fname

def getRecordingFilename(d, settings, suffix = None):
    fdir = os.path.join(d, 'croprot/')

    if not os.path.exists(fdir):
        return None

    # Check / preprocess settings
    settings = checkSettings(settings)

    # Extract relevant parameters from settings
    if 'algorithm' not in settings:
        raise Exception('Algorithm not specified.')
    else:
        algorithm = settings['algorithm']

    # Find the corresponding filename
    if suffix is None:
        suffix = '_{}_abs_filt_interp_mvmt_noborder.npy'.format(settings['algorithm'])

    fname = [os.path.join(fdir, x) for x in os.listdir(fdir) if \
             x.endswith(suffix)]
    fname = None if len(fname) != 1 else fname[0]

    # Done!
    return fname

def findUnprocessedRecordings(rootpath, overwrite=True):
    raise NotImplementedError()

    dirs = []
    for x in os.listdir(rootpath):
        fx = os.path.join(rootpath, x)
        # Only proceed if this is a recording directory
        if isRecordingDir(fx):
            # Obtain cropped/rotated filename
            fname = getRecordingFilename(fx)
            if fname is not None:
                dirs.append(fname)
    return dirs

def isRecordingDir(dirpath):
    return os.path.exists(os.path.join(dirpath, 'raw')) or os.path.exists(os.path.join(dirpath, 'ufmf'))

# Bring angle closest to 0 (e.g. 350 > -10, -350 > 10)
def thetaToHalfCircle(x):
    while x > np.pi:
        x -= 2 * np.pi
    while x < -np.pi:
        x += 2 * np.pi
    return x

# =====================================================================================
# Mathematical functions for wavelet transform
# =====================================================================================

# morletConjFT is used by fastWavelet_morlet_convolution_parallel to find
# the Morlet wavelet transform resulting from a time series
@jit(nopython=True)
def morletConjFT(w, omega0):
    return np.power( np.pi, -1/4 ) * np.exp( -0.5 * np.power(w-omega0, 2) );


# fastWavelet_morlet_convolution_parallel finds the Morlet wavelet transform
# resulting from a time series
#
#   Input variables:
#
#       x -> 1d array of projection values to transform
#       f -> center bands of wavelet frequency channels (Hz)
#       omega0 -> dimensionless Morlet wavelet parameter
#       dt -> sampling time (seconds)
#
#
#   Output variables:
#
#       amp -> wavelet amplitudes (N x (pcaModes*numPeriods) )
#       W -> wavelet coefficients (complex-valued)
#
#
# (C) Gordon J. Berman, 2014
#     Princeton University

def fastWavelet_morlet_convolution_parallel(x, f, omega0, dt):
    N = x.shape[0]
    L = f.shape[0]
    amp = np.zeros((L, N))

    test = None
    if np.mod(N, 2) == 1:
       x = np.append(x, [0])
       N = N + 1;
       test = True
    else:
       test = False

    if len(x.shape) == 1:
        x = np.asmatrix(x)

    if x.shape[1] == 1:
        x = x.T

    x = np.hstack((np.zeros((1,int(N / 2))), x, np.zeros((1,int(N / 2)))))
    M = N
    N = x.shape[1]

    scales = (omega0 + np.sqrt(2 + omega0 ** 2)) / (4 * np.pi * f)
    Omegavals = 2 * np.pi * np.arange(-N / 2, N / 2) / (N * dt)

    xHat = np.fft.fft(x)
    xHat = np.fft.fftshift(xHat)

    idx = None
    if test:
        idx = np.arange(M / 2, M / 2 + M - 1, dtype=int)
    else:
        idx = np.arange(M / 2, M / 2 + M, dtype=int)

    returnW = True

    test2 = None
    if returnW:
        W = np.zeros(amp.shape);
        test2 = True;
    else:
        test2 = False;

    for i in range(L):
        m = morletConjFT(- Omegavals * scales[i], omega0)
        q = np.fft.ifft(m * xHat) * np.sqrt(scales[i])
        q = q[0,idx]
        amp[i, :] = np.abs(q) * np.power(np.pi, -0.25) * \
                    np.exp(0.25 * np.power(omega0 - np.sqrt(omega0 ** 2 + 2), 2)) / np.sqrt(2 * scales[i])

        if returnW:
            W[i, :] = q

    return amp, W


# findWavelets finds the wavelet transforms resulting from a time series
#
#   Input variables:
#
#       projections -> N x d array of projection values
#       numModes -> # of transforms to find
#       parameters -> struct containing non-default choices for parameters
#
#   Output variables:
#
#       amplitudes -> wavelet amplitudes (N x (pcaModes*numPeriods) )
#       f -> frequencies used in wavelet transforms (Hz)
#
# (C) Gordon J. Berman, 2014
#     Princeton University
#  MODIFIED: Andrew Gordus, 2015, The Rockefeller University

def WaveletCalc(projections, numModes=None, parameters=None):
    # ...
    d1, d2 = projections.shape
    if d2 > d1:
        projections = projections.T

    # ...
    L = projections[1, :].shape[0]
    if numModes is None:
        numModes = L;
    else:
        if numModes > L:
            numModes = L

    # Extract parameters
    omega0 = parameters['omega0']
    numPeriods = parameters['numPeriods']
    dt = 1 / parameters['samplingFreq']
    minT = 1 / parameters['maxF']
    maxT = 1 / parameters['minF']

    Ts = minT * np.power(2, np.arange(0, numPeriods) * np.log(maxT / minT) / (np.log(2) * (numPeriods - 1)))
    f = np.flip(1 / Ts)
    N = projections[:, 0].shape[0]

    if parameters['stack']:
        amplitudes = np.zeros((N, numModes * numPeriods))
        for i in range(numModes):
            temp, W = fastWavelet_morlet_convolution_parallel(
                projections[:, i], f, omega0, dt)
            temp = np.fliplr(temp)
            temp = temp / np.max(temp)
            # import pdb; pdb.set_trace()
            amplitudes[:, np.arange(i * numPeriods, (i + 1) * numPeriods, dtype=int)] = temp.T
    else:
        raise NotImplementedError("Only 'stack' mode is supported.")
        #amplitudes = zeros(N,numModes,numPeriods);
        #for i=1:numModes:
        #    temp = ...
        #        fastWavelet_morlet_convolution_parallel_ag(...
        #        projections(:,i),f,omega0,dt)';
        #    temp = temp./max(temp(:));
        #    amplitudes(:,i,:) = temp;

    return amplitudes, f

# =====================================================================================
# Functions for orienting the spider posture data
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

@jit(nopython=True, nogil=True)
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

def alignSpider(datarel):
    # Find orientation of spider
    v = datarel[:, 0, :] - datarel[:, 1, :]
    theta = np.arctan2(v[:, 0], v[:, 1])

    # Apply the rotation
    applyRotation(theta, datarel)

    return theta

# =====================================================================================
# Settings helper function
# =====================================================================================

def checkSettings(settings):
    # Check if overwrite flag is specified, and set to default otherwise
    if not 'overwrite' in settings:
        settings['overwrite'] = False

    # Check the joint specifications
    if 'joints' in settings:
        if isinstance(settings['joints'], list):
            settings['joints'] = ''.join(['1' if i in settings['joints'] else '0' for i in range(26)])
    else:
        settings['joints'] = ''.join('1' for i in range(26))

    # Check if algorithm is specified, if not, default to LEAP
    if 'algorithm' not in settings:
        settings['algorithm'] = 'leap'

    # Check if PCA flag is specified
    if 'use_pca' not in settings:
        settings['use_pca'] = 'no-pca'
    elif settings['use_pca'] not in ['no-pca','pca','nmf']:
        raise Exception('use_pca not any of: "no-pca", "pca", "nmf"')

    # Check if coordinate style is specified? Currently one of: 'euclidean', 'polar'
    if 'coords' not in settings:
        settings['coords'] = 'polar'

    # Check if absolute position should be included
    if 'include_absolute_position' not in settings:
        settings['include_absolute_position'] = 'with-abspos'

    if 'include_velocities' not in settings:
        settings['include_velocities'] = 'with-vel'

    if 'numPeriods' not in settings:
        settings['numPeriods'] = 20

    if 'histogram_normalize' not in settings:
        settings['histogram_normalize'] = True

    return settings

# =====================================================================================
# Apply wavelet transform with the appropriate settings
# =====================================================================================

def runWaveletTransform(dir, settings, loadWaveletIfExists = True):

    # Preprocess settings
    settings = checkSettings(settings)

    # Get output filename
    fnameWav = getWaveletFilename(dir, settings)
    fnameFreq = fnameWav.replace('.npy', '') + '.freq.npy'

    # Has the wavelet already been computed?
    if os.path.exists(fnameWav) and not settings['overwrite']:
        if loadWaveletIfExists:
            print("Loading wavelet transform...")
            # Load data
            try:
                amplitudes = np.load(fnameWav)
                f = np.load(fnameFreq)
                # Return
                return amplitudes, f
            except Exception as e:
                logging.info('Error while loading existing wavelet file. File corrupted, recomputing...')
        else:
            # Still check if file is valid by opening it in memory-mapped mode
            try:
                amplitudes = np.load(fnameWav, mmap_mode='r')
                del amplitudes
                # If the user did not request loading the wavelet if it exists, just return null
                return None, None
            except Exception as e:
                logging.info('Error while loading existing wavelet file. File corrupted, recomputing...')

    # obtain filename of wavelet output given these settings
    # -- We run wavelet on all the data, to avoid interruptions, then subset to movement/noborder later
    fnameSuffix = '_{}_abs_filt_interp.npy'.format(settings['algorithm'] if settings['algorithm'] in ['leap','dlc'] else 'dlc')
    fname = getRecordingFilename(dir, settings, fnameSuffix)

    # Load data
    data = np.load(fname)[:,:,0:2]

    # Center the data
    xy = np.mean(data[:, [0, 1], :], axis=1)
    datarel = data - np.repeat(xy[:, np.newaxis, :], 26, axis=1)

    # Sanity Check: Check there are no NaN's left
    assert(not np.any(np.isnan(datarel)))

    # Make alignment of spider fixed
    theta = alignSpider(datarel)

    # Fix tracking data alignment?
    if settings['coords'] == "euclidean-midline":
        jointIDs = np.argwhere(np.array([1 if x == '1' else 0 for x in list(settings['joints'])])).T[0]
        import pipeline.python.behavioral_motifs.run_wavelet_cluster_rawmvmt as rwcr
        datarel = np.moveaxis(datarel, 2, 1)
        datarel = rwcr.alignTrackingByMidline(datarel, jointIDs)
        datarel = np.moveaxis(datarel, 1, 2)

    # Save the interpolated, aligned data
    np.save(fnameWav.replace('.npy', '') + '_{}.coords.npy'.format(settings['algorithm']), datarel)

    # Now compute relative joint position w.r.t. connecting joints
    datarelRel = datarel.copy()[:,:,0:2]
    for j in JOINT_PARTNERS.keys():
        datarelRel[:, j, :] -= datarel[:, JOINT_PARTNERS[j], 0:2]

        # Rotate such that the joint partner is fixed in orientation
        p2 = datarel[:, JOINT_PARTNERS[j], 0:2]
        p1 = datarel[:, JOINT_PARTNERS[JOINT_PARTNERS[j]], 0:2]
        v = p2 - p1
        theta2 = np.arctan2(v[:, 0], v[:, 1])
        applyRotation(theta2, datarelRel[:, j, :])

        # Switch to polar coordinate system if settings indicate it
        if settings['coords'] == 'polar':
            v = datarelRel[:, j, :] - p2
            theta3 = np.arctan2(v[:, 0], v[:, 1])
            length= np.linalg.norm(v, axis=1)
            datarelRel[:, j, 0] = theta3
            datarelRel[:, j, 1] = length

    datarel = datarelRel

    # Flatten 3rd dimension, i.e. X and Y next to each other
    # (matrix columns: rel X pos (26), rel Y pos (26))
    datarelflat = np.hstack((datarel[:, :, 0], datarel[:, :, 1]))

    _col_offset = 0
    if not 'include_velocities' in settings or settings['include_velocities'] == 'with-vel':
        print('including velocities')
        _col_offset += 3

        # Compute velocity signal for center of spider (1st derivative of position, separately for X and Y)
        # -- Compute velocity w.r.t. a frame delta of 5 to avoid jitter dominating the true signal
        vel = xy - np.roll(xy, 5, axis=0)
        # -- Clip to ~2 cm/s max speed, which seems to be an upper limit of true velocities
        vel = np.clip(vel, -40, 40)

        # Add velocity signals to flattened array
        # (matrix columns: vel X (1), vel Y(1), rel X pos (26), rel Y pos (26))
        datarelflat = np.hstack((vel, datarelflat))

        # Add angular velocity (1st derivative of angle)
        #angvel = np.pad(np.diff(theta, axis=0), (1, 0), mode='constant', constant_values=0)
        angvel = np.vectorize(thetaToHalfCircle)(theta - np.roll(theta, 12, axis=0))

        # If the velocity is outside -180-+180, bring it within this range
        angvel = np.vectorize(thetaToHalfCircle)(angvel)

        # Add angular velocity signal
        # (matrix columns: angular vel (1), vel X (1), vel Y(1), rel X pos (26), rel Y pos (26))
        datarelflat = np.hstack((angvel[:, np.newaxis], datarelflat))

    # Optional: Add absolute position
    if settings['include_absolute_position'] == 'with-abspos':
        print('including absolute position')
        _col_offset += 3
        datarelflat = np.hstack((xy, datarelflat))
        ang = np.vectorize(thetaToHalfCircle)(theta)
        datarelflat = np.hstack((ang[:, np.newaxis], datarelflat))

    # Sanity Check: Check there are no NaN's left
    assert (not np.any(np.isnan(datarelflat)))

    # Now if we instead are opting for 'pca coordinates', optioanly keep vel/abspos data, but remove all leg coordinates
    if settings['algorithm'] == 'rawpca':
        #   - Get name of 200x200 cropped/rotated image
        fnameCroprot = getRecordingFilename(dir, settings, '_img.npy')
        #   - Get the name of the PCA output (so it does not have to be recomputed unnecessarily)
        fnameCroprotPCA = fnameCroprot.replace('_img.npy','') + '_img_pca.npy'
        #   - Load array (memory-mapped)
        imgCroprot = np.memmap(fnameCroprot, mode='r', dtype=np.uint8)
        N = int(imgCroprot.shape[0] / (200 * 200))
        del imgCroprot
        imgCroprot = np.memmap(fnameCroprot, mode='r', dtype=np.uint8, shape=(N, 200, 200))
        #   - Open the array
        NUM_PCA_COMPONENTS = 50
        N = imgCroprot.shape[0]
        if os.path.exists(fnameCroprotPCA):
            imgPCA = np.memmap(fnameCroprotPCA, mode='r+', dtype=np.float32, shape=(N, NUM_PCA_COMPONENTS))
        else:
            imgPCA = np.memmap(fnameCroprotPCA, mode='w+', dtype=np.float32, shape=(N, NUM_PCA_COMPONENTS))
        #   - Now perform PCA on this image matrix (for now, choose a fixed number of components)
        DOWNSAMPLE, SUBSET_PCAFIT = 1, 10
        CROP_START, CROP_END = 40, 160
        pca = PCA(n_components=NUM_PCA_COMPONENTS, svd_solver='full')
        logging.info('Generating PCA dataset: {}'.format(fnameCroprotPCA))
        _X  = imgCroprot[::SUBSET_PCAFIT, CROP_START:CROP_END:DOWNSAMPLE, CROP_START:CROP_END:DOWNSAMPLE].reshape(
            int(imgCroprot.shape[0]/SUBSET_PCAFIT), int(((CROP_END - CROP_START) / DOWNSAMPLE) ** 2)).astype(np.float32)
        logging.info('Fitting PCA: {}'.format(fnameCroprotPCA))
        pca.fit(_X)
        # Save PCA model for future reference
        fnameCroprotPCAmodel = fnameCroprotPCA.replace('.npy','') + '.model.pickle'
        jl.dump(pca, fnameCroprotPCAmodel)
        logging.info('Applying PCA transform...')
        # Transform data
        for c in tqdm(range(0, imgCroprot.shape[0], 50000),
                total=int(np.ceil(imgCroprot.shape[0] / 50000.0)), desc='PCA Chunks Transformed'):
            imgPCA[c:(c+50000), :] = pca.transform(imgCroprot[c:(c+50000),CROP_START:CROP_END:DOWNSAMPLE,CROP_START:CROP_END:DOWNSAMPLE].reshape(
                imgCroprot[c:(c+50000),:,:].shape[0], int( ((CROP_END - CROP_START) / DOWNSAMPLE)**2 )))
            gc.collect()
        # Save PCA coordinates
        fnameCroprotPCAcoords = fnameCroprotPCA.replace('.npy','') + '.pcacoords.npy'
        np.save(fnameCroprotPCAcoords, imgPCA)

        # Update status
        logging.info('Finished PCA transform...')

        # Update the wavelet transform array
        if _col_offset > 0:
            #   - First remove the limb coordinates from 'datarelflat'
            datarelflat = datarelflat[:,0:_col_offset]
            #   - Now add the PCA coordinates
            datarelflat = np.hstack( (datarelflat, imgPCA) )
        else:
            datarelflat = imgPCA.copy()
        
        #   - Close the memmap'ed files
        del imgCroprot
        del imgPCA

    # Z score all values
    # -- Convert to 64 bit float b/c this centering is more accurate with it:
    dataZscore = datarelflat.astype(np.double)
    # -- Center all columns
    dataZscore = dataZscore - np.repeat(np.nanmean(dataZscore, axis=0)[np.newaxis, :], dataZscore.shape[0], axis=0)
    # -- Divide by standard deviation
    dataZscore /= np.repeat(np.nanstd(dataZscore, axis=0)[np.newaxis, :], dataZscore.shape[0], axis=0)
    # -- Convert to 32 bit b/c we no longer need excessive precision
    dataZscore = dataZscore.astype(np.float32)

    # Now subset only the leg joints we want
    jointsLegs = [x for x in JOINTS_LEGS if settings['joints'][x] == '1']

    # Leg positions
    # To-do: Option to take this data from either Z-scored or other transform of data
    if settings['algorithm'] == 'rawpca':
        dataLegs = dataZscore[:, _col_offset:]
    else:
        dataLegs = dataZscore[:, (_col_offset + np.array(jointsLegs)).tolist() + (_col_offset + 26 + np.array(jointsLegs)).tolist()]

    # X, Y, theta velocities
    # To-do: Option to take this data from either Z-scored or other transform of data
    dataThorax = dataZscore[:, 0:_col_offset]

    # Get leg vars
    if not 'legvars' in settings:
        legvars = {}
        legvars['samplingFreq'] = 50;  # fps
        legvars['omega0'] = 5;
        legvars['numPeriods'] = settings['numPeriods']
        legvars['maxF'] = legvars['samplingFreq'] / 2;
        legvars['minF'] = 0.1;  # hz
        legvars['stack'] = True;
        legvars['numProcessors'] = 4;
        settings['legvars'] = legvars
    else:
        legvars = settings['legvars']

    # Get thorax vars
    if not 'thoraxvars' in settings:
        thoraxvars = {}
        thoraxvars['samplingFreq'] = 50;  # fps
        thoraxvars['omega0'] = 5;
        thoraxvars['numPeriods'] = settings['numPeriods']
        thoraxvars['maxF'] = 1;
        thoraxvars['minF'] = 0.04;  # hz
        thoraxvars['stack'] = True;
        thoraxvars['numProcessors'] = 4;
        settings['thoraxvars'] = thoraxvars
    else:
        thoraxvars = settings['thoraxvars']

    # Perform wavelet transform
    print("Running wavelet transform...")

    # Do wavelet transform for thorax variables (X translation, Y translation, Rotation)
    if dataThorax.shape[1] > 0 and dataLegs.shape[1] > 0:
        amplitudesThorax, fThorax = WaveletCalc(dataThorax, dataThorax.shape[1], parameters=thoraxvars)
        # Do wavelet transform for legs
        amplitudesLegs, fLegs = WaveletCalc(dataLegs, dataLegs.shape[1], parameters=legvars)
        # Append matrices
        amplitudes = np.hstack((amplitudesThorax, amplitudesLegs))
        f = np.hstack((fThorax, fLegs))
    elif dataLegs.shape[1] > 0:
        # Do wavelet transform for legs
        amplitudes, f = WaveletCalc(dataLegs, dataLegs.shape[1], parameters=legvars)
    elif dataThorax.shape[1] > 0:
        # Do wavelet transform for legs
        amplitudes, f = WaveletCalc(dataThorax, dataThorax.shape[1], parameters=thoraxvars)
    elif dataLegs.shape[1] > 0:
        raise Exception("No variables selected for wavelet transform.")

    # Normalize so average power is the same for all limbs
    assert(thoraxvars['numPeriods'] == legvars['numPeriods'])
    numperiods = thoraxvars['numPeriods']
    for spstart in range(0, amplitudes.shape[1], numperiods):
        amplitudes[:,spstart:(spstart+numperiods)] /= np.nanmean(amplitudes[:,spstart:(spstart+numperiods)])

    # Subset data back to original movement/noborder subset
    idxSS = np.load(fname.replace('_filt_interp.npy', '_filt_interp_mvmt_noborder.idx.npy'))
    amplitudes = amplitudes[idxSS,:]
    if dataThorax.shape[0] == idxSS.shape[0]:
        dataThorax = dataThorax[idxSS,:]
    if dataLegs.shape[0] == idxSS.shape[0]:
        dataLegs = dataLegs[idxSS, :]

    # Save leg and thorax data
    scipy.io.savemat(fnameWav.replace('.npy','') + '.wavelet_input.mat', {'thorax': dataThorax, 'legs': dataLegs})

    # Save
    np.save(fnameFreq, f)
    np.save(fnameWav, amplitudes)

    # Plot
    plotOverview(dir, settings, amplitudes, f)

    # Return data
    return amplitudes, f

def runWaveletTransformSafe(dir, settings, loadWaveletIfExists = True):
    try:
        runWaveletTransform(dir, settings, loadWaveletIfExists)
    except Exception as e:
        logging.warning(dir + '\n' + str(e))

# =====================================================================================
# List all valid, full-web recordings
# =====================================================================================

def listAvailableRecordings(datadir = 'Z:/behavior/'):
    fnames = []
    for d in os.listdir(datadir):
        fnameRecordingMetadata = os.path.join(*[datadir, d, 'recording.json'])
        if os.path.exists(fnameRecordingMetadata):
            with open(fnameRecordingMetadata, 'r') as f:
                txt = f.read()
                try:
                    fileMeta = json.loads(txt)
                    # Make sure this web has been manually evaluated as complete and successfully tracked
                    if fileMeta['web_complete'] == True and fileMeta['tracking_successful'] == True:
                        # Make sure the input file exists
                        dpath = os.path.join(datadir, d)
                        fnameRec = getRecordingFilename(dpath, {'algorithm': 'dlc'})
                        if os.path.exists(fnameRec):
                            fnames.append( dpath )
                except Exception as e:
                    print(d, e)
                    pass
    return fnames

# =====================================================================================
# Plot parameters
# =====================================================================================

def plotOverview(dir, settings, ampl, f):
    # Preprocess settings
    settings = checkSettings(settings)

    # Get output filename
    fnameWav = getWaveletFilename(dir, settings)
    fnameOut = fnameWav.replace('.npy', '') + '.png'

    # Already exists?
    if os.path.exists(fnameOut) and not settings['overwrite']:
        return

    # Plot amplitudes across multiple bands for visual inspection
    plt.ioff()
    fig, axes = plt.subplots(10, 1, facecolor='w', figsize=(18, 26))

    # Number of timepoints to display per line
    TP_PER_LINE = int(np.ceil(ampl.shape[0] / len(axes)))

    # Plot each line
    for i in range(len(axes)):
        axes[i].imshow(ampl[(i * TP_PER_LINE):(i + 1) * TP_PER_LINE, :].T, interpolation='nearest', aspect='auto')

    # Show figure
    fig.savefig(fnameOut)

# =====================================================================================
# Test
# =====================================================================================

def processBatch():
    configs = []

    fnames = listAvailableRecordings()

    j1 = [6, 10, 18, 22, 9, 13, 21, 25]
    j2 = [10, 22, 13, 25]
    j3 = []

    for numPeriods in [20, 40]:
        for algorithm in ['dlc', ]:  # 'rawpca']:
            for coords in ['polar', 'euclidean']:
                for include_absolute_position in ['no-abspos', 'with-abspos']:
                    for include_vel in ['with-vel', ]:  # ['no-vel','with-vel']:
                        for joints in [j3, j2, j1]:
                            for fname in fnames:
                                configs.append((fname, {
                                    'algorithm': algorithm,
                                    # Include only front legs
                                    'joints': joints,
                                    # Overwrite?
                                    'overwrite': True,
                                    # Use polar coordinates w.r.t. previous limb?
                                    'coords': coords,
                                    'include_absolute_position': include_absolute_position,
                                    'include_velocities': include_vel,
                                    'use_pca': 'no-pca',
                                    'numPeriods': numPeriods,
                                    'parallel': False
                                }, False))

    # Process files/configurations in randomized order
    np.random.shuffle(configs)

    # Run transform
    while True:
        try:
            jl.Parallel(n_jobs=25)(jl.delayed(runWaveletTransformSafe)(*config) for config in tqdm(configs))
            # Only break out of this loop when no errors occurred
            break
        except Exception as e:
            print(e)
            # Resumes processing autoamtically due to while loop...


if __name__ == "__main__":

    configs = []

    fnames = [x for x in listAvailableRecordings() if '6-3-19-e' in x]

    j1 = [6, 10, 18, 22, 9, 13, 21, 25]
    j2 = [10, 22, 13, 25]
    j3 = []

    for numPeriods in [20]: #, 40]:
        for algorithm in ['dlc',]: # 'rawpca']:
            for coords in ['euclidean',]: # 'polar'
                for include_absolute_position in ['no-abspos',]:
                    for include_vel in ['with-vel','no-vel']: #['no-vel','with-vel']:
                        for joints in [j3,j2,j1]:
                            for fname in fnames:
                                configs.append((fname, {
                                    'algorithm': algorithm,
                                    # Include only front legs
                                    'joints': joints,
                                    # Overwrite?
                                    'overwrite': True,
                                    # Use polar coordinates w.r.t. previous limb?
                                    'coords': coords,
                                    'include_absolute_position': include_absolute_position,
                                    'include_velocities': include_vel,
                                    'use_pca': 'no-pca',
                                    'numPeriods': numPeriods,
                                    'parallel': False,
                                    'legvars': {
                                        'samplingFreq': 50,
                                        'omega0': 5,
                                        'numPeriods': numPeriods,
                                        'maxF': 25,
                                        'minF': 1,
                                        'stack': True,
                                        'numProcessors': 4
                                    }
                                }, False))

    # Process files/configurations in randomized order
    np.random.shuffle(configs)

    # Run transform
    while True:
        try:
            njobs = 1 # 25
            jl.Parallel(n_jobs=njobs)(jl.delayed(runWaveletTransformSafe)(*config) for config in tqdm(configs))
            # Only break out of this loop when no errors occurred
            break
        except Exception as e:
            print(e)
            # Resumes processing automatically due to while loop...


